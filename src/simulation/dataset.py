import os
import sys
import csv
import json
import itertools
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner, extract_fault_info
from src.simulation.filter import remove_low_frequency_components
from src.estimator.cla_estimator import ConsumerLoadPremises, ClusterLoadAllocationEstimator
from src.estimator.time_adjusted_cla_estimator import TimeAdjustedCLAEstimator
from src.transient.events import (
    SingleEquipmentSwitchEvent,
    SingleLineFaultEvent,
    EquipmentEquipmentCoEvent,
    EquipmentLineFaultCoEvent
)


def get_all_108_coevents(target_line: str = "feeder1_head"):
    """
    Generates the complete 108 unique co-event space for N_L = 8 consumer load models
    and N_F = 10 phase-specific fault configurations.

    Pairs:
    - 28 Load-Load pairs (C(8, 2))
    - 80 Load-Fault pairs (8 * 10: 24 LG, 24 LL, 24 LLG, 8 LLL)
    Total: 28 + 80 = 108 unique co-events.
    """
    equipment_types = [
        "ac_motor", "dc_motor_inverter", "microwave", "induction_plate",
        "compressor", "audio_amplifier", "ups", "industrial_fan"
    ]

    fault_configs = [
        ("LG", (0,), "AG"),
        ("LG", (1,), "BG"),
        ("LG", (2,), "CG"),
        ("LL", (0, 1), "AB"),
        ("LL", (1, 2), "BC"),
        ("LL", (2, 0), "CA"),
        ("LLG", (0, 1), "ABG"),
        ("LLG", (1, 2), "BCG"),
        ("LLG", (2, 0), "CAG"),
        ("LLL", (0, 1, 2), "ABC")
    ]

    co_events = []

    # 1. Load-Load Pairs (28)
    for eq1_type, eq2_type in itertools.combinations(equipment_types, 2):
        ev1 = SingleEquipmentSwitchEvent(eq1_type, 0.02, 0.04, target_line, {})
        ev2 = SingleEquipmentSwitchEvent(eq2_type, 0.02, 0.04, target_line, {})
        co_events.append(("load_load", EquipmentEquipmentCoEvent(ev1, ev2)))

    # 2. Load-Fault Pairs (80)
    for eq_type, (f_type, phases, f_name) in itertools.product(equipment_types, fault_configs):
        ev1 = SingleEquipmentSwitchEvent(eq_type, 0.02, 0.04, target_line, {})
        ev2 = SingleLineFaultEvent(f_type, 0.02, 0.04, target_line, phases, 0.05, {"config_id": f_name})
        co_events.append(("load_fault", EquipmentLineFaultCoEvent(ev1, ev2)))

    return co_events


def extract_load_source(co_ev):
    sources = []
    for ev in [co_ev.event_1, co_ev.event_2]:
        if getattr(ev, "event_class", "") == "equipment_switch":
            target = getattr(ev, "target", "feeder1_head")
            sources.append({"bus": target, "line": f"line_{target}"})
    return json.dumps(sources) if sources else ""


def _process_coevent_worker(task_args):
    """
    ProcessPoolExecutor worker function that executes a single co-event operation in an isolated process.
    OpenDSS solves for network parameters under line fault or steady-state conditions, and ATP simulates
    consumer load transients and transformer transients.
    """
    p_idx, pair_cat, co_ev, use_baseline_transformers, time_offset = task_args
    runner = CoSimulationRunner()
    runner.initialize_plant_session(use_baseline_transformers=use_baseline_transformers, seed=42 + p_idx)

    ev1 = co_ev.event_1
    ev2 = getattr(co_ev, "event_2", None)
    if time_offset > 0 and hasattr(co_ev, "with_time_shift"):
        co_ev_effective = co_ev.with_time_shift(time_offset)
        ev2 = co_ev_effective.event_2
    else:
        co_ev_effective = co_ev

    f_id = (p_idx % 3) + 1
    m_id = f"trans{f_id}_lv_boundary_consumer_unit"

    # 1. Combined co-event simulation (OpenDSS solves network parameters; ATP simulates load transients)
    sim_co = runner.run_simulation(
        events=[co_ev_effective],
        use_baseline_transformers=use_baseline_transformers,
        include_load_event=True,
        include_fault_event=True,
        scenario_id=f"coev_{p_idx}_{f_id}",
        seed=42 + p_idx,
        reinitialize_plant=False
    )
    unit_co = sim_co.processed_consumer_units.get(m_id, {})
    v_co = np.array(unit_co.get("raw_voltage", np.zeros((1000, 3))))
    i_co = np.array(unit_co.get("raw_current", np.zeros((1000, 3))))

    # 2. Constituent event 1 simulation
    sim_1 = runner.run_simulation(
        events=[ev1],
        use_baseline_transformers=use_baseline_transformers,
        include_load_event=True,
        include_fault_event=(getattr(ev1, "event_class", "") == "line_fault"),
        scenario_id=f"ev1_{p_idx}_{f_id}",
        seed=42 + p_idx,
        reinitialize_plant=False
    )
    unit_1 = sim_1.processed_consumer_units.get(m_id, {})
    v1_sig = np.array(unit_1.get("raw_voltage", np.zeros((1000, 3))))
    i1_sig = np.array(unit_1.get("raw_current", np.zeros((1000, 3))))

    # 3. Constituent event 2 simulation
    sim_2 = runner.run_simulation(
        events=[ev2],
        use_baseline_transformers=use_baseline_transformers,
        include_load_event=True,
        include_fault_event=(getattr(ev2, "event_class", "") == "line_fault"),
        scenario_id=f"ev2_{p_idx}_{f_id}",
        seed=42 + p_idx,
        reinitialize_plant=False
    )
    unit_2 = sim_2.processed_consumer_units.get(m_id, {})
    v2_sig = np.array(unit_2.get("raw_voltage", np.zeros((1000, 3))))
    i2_sig = np.array(unit_2.get("raw_current", np.zeros((1000, 3))))

    t_s = sim_co.time_s if len(sim_co.time_s) > 0 else np.linspace(0.0, 0.1, 1000)

    # High-pass filter fundamental components
    v1_hp = remove_low_frequency_components(v1_sig)
    i1_hp = remove_low_frequency_components(i1_sig)
    v2_hp = remove_low_frequency_components(v2_sig)
    i2_hp = remove_low_frequency_components(i2_sig)

    v_comp = v1_hp + v2_hp
    i_comp = i1_hp + i2_hp

    v_co_hp = remove_low_frequency_components(v_co)
    i_co_hp = remove_low_frequency_components(i_co)

    res_v = v_co_hp - v_comp
    res_i = i_co_hp - i_comp

    v_mag = float(np.max(np.abs(res_v)))
    i_mag = float(np.max(np.abs(res_i)))

    row_data = {
        "p_idx": p_idx,
        "f_id": f_id,
        "load_source": extract_load_source(co_ev_effective),
        "fault_info": extract_fault_info(co_ev_effective),
        "gt_event_1_equipment_type": getattr(ev1, "equipment_type", ""),
        "gt_event_1_fault_type": getattr(ev1, "fault_type", ""),
        "gt_event_1_start_timestamp_s": float(ev1.start_time_s),
        "gt_event_2_equipment_type": getattr(ev2, "equipment_type", ""),
        "gt_event_2_fault_type": getattr(ev2, "fault_type", ""),
        "gt_event_2_start_timestamp_s": float(ev2.start_time_s),
        "gt_time_offset_s": float(time_offset),
        "obs_coevent_time": json.dumps(t_s.tolist()),
        "obs_coevent_v": json.dumps([v_co[:, 0].tolist(), v_co[:, 1].tolist(), v_co[:, 2].tolist()]),
        "obs_coevent_i": json.dumps([i_co[:, 0].tolist(), i_co[:, 1].tolist(), i_co[:, 2].tolist()]),

        "obs_single_event_1_v_phase_a": json.dumps(v1_hp[:, 0].tolist()),
        "obs_single_event_1_v_phase_b": json.dumps(v1_hp[:, 1].tolist()),
        "obs_single_event_1_v_phase_c": json.dumps(v1_hp[:, 2].tolist()),
        "obs_single_event_1_i_phase_a": json.dumps(i1_hp[:, 0].tolist()),
        "obs_single_event_1_i_phase_b": json.dumps(i1_hp[:, 1].tolist()),
        "obs_single_event_1_i_phase_c": json.dumps(i1_hp[:, 2].tolist()),

        "obs_single_event_2_v_phase_a": json.dumps(v2_hp[:, 0].tolist()),
        "obs_single_event_2_v_phase_b": json.dumps(v2_hp[:, 1].tolist()),
        "obs_single_event_2_v_phase_c": json.dumps(v2_hp[:, 2].tolist()),
        "obs_single_event_2_i_phase_a": json.dumps(i2_hp[:, 0].tolist()),
        "obs_single_event_2_i_phase_b": json.dumps(i2_hp[:, 1].tolist()),
        "obs_single_event_2_i_phase_c": json.dumps(i2_hp[:, 2].tolist()),

        "obs_composed_event_v_phase_a": json.dumps(v_comp[:, 0].tolist()),
        "obs_composed_event_v_phase_b": json.dumps(v_comp[:, 1].tolist()),
        "obs_composed_event_v_phase_c": json.dumps(v_comp[:, 2].tolist()),
        "obs_composed_event_i_phase_a": json.dumps(i_comp[:, 0].tolist()),
        "obs_composed_event_i_phase_b": json.dumps(i_comp[:, 1].tolist()),
        "obs_composed_event_i_phase_c": json.dumps(i_comp[:, 2].tolist()),

        "obs_residual_v": json.dumps([res_v[:, 0].tolist(), res_v[:, 1].tolist(), res_v[:, 2].tolist()]),
        "obs_residual_i": json.dumps([res_i[:, 0].tolist(), res_i[:, 1].tolist(), res_i[:, 2].tolist()]),

        "obs_residual_v_phase_a": json.dumps(res_v[:, 0].tolist()),
        "obs_residual_v_phase_b": json.dumps(res_v[:, 1].tolist()),
        "obs_residual_v_phase_c": json.dumps(res_v[:, 2].tolist()),
        "obs_residual_i_phase_a": json.dumps(res_i[:, 0].tolist()),
        "obs_residual_i_phase_b": json.dumps(res_i[:, 1].tolist()),
        "obs_residual_i_phase_c": json.dumps(res_i[:, 2].tolist()),
        "residual_voltage_magnitude": v_mag,
        "residual_current_magnitude": i_mag
    }

    if not use_baseline_transformers:
        row_data["gt_feeder_id"] = f"feeder_{f_id}"

    return row_data


def process_dataset_coevents_in_batches(all_108_pairs, use_baseline_transformers: bool, is_dataset_3: bool = False, dataset_name: str = "Dataset", batch_size: int = 6, max_workers: int = 4):
    """
    Processes 108 co-events in batches of 6 operations using ProcessPoolExecutor.
    108 co-events / 6 per batch = 18 batch operations per dataset.
    Logs statements when each batch completes and when dataset generation starts and completes.
    """
    tasks = []
    for p_idx, (pair_cat, co_ev) in enumerate(all_108_pairs):
        time_offset = (0.01 if p_idx % 2 == 1 else 0.0) if is_dataset_3 else 0.0
        tasks.append((p_idx, pair_cat, co_ev, use_baseline_transformers, time_offset))

    num_batches = int(np.ceil(len(tasks) / float(batch_size)))
    print(f"INFO: Starting {dataset_name} generation ({len(tasks)} co-events across {num_batches} batches, {batch_size} operations per batch)...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for b_idx in range(num_batches):
            batch_tasks = tasks[b_idx * batch_size : (b_idx + 1) * batch_size]
            print(f"INFO: Submitting Batch {b_idx + 1}/{num_batches} for {dataset_name} ({len(batch_tasks)} co-events)...")
            batch_results = list(executor.map(_process_coevent_worker, batch_tasks))
            results.extend(batch_results)
            print(f"INFO: Completed Batch {b_idx + 1}/{num_batches} for {dataset_name}.")

    # Sort results to maintain original co-event order
    results.sort(key=lambda r: r["p_idx"])
    # Clean temporary process ordering key
    for r in results:
        r.pop("p_idx", None)
        r.pop("f_id", None)

    print(f"INFO: Completed {dataset_name} generation ({len(results)} total co-events).")
    return results


def generate_experiments_dataset(write_to_disk: bool = True):
    """
    Orchestrates dataset generation for Dataset 1 (Cluster Load Allocation energy estimation),
    Dataset 2 (108 unique co-events observability), Dataset 3 (108 unique co-events time shift),
    and Dataset 4 (108 unique co-events transformer spec effect) using CoSimulationRunner functions ONLY.
    """
    print("INFO: Starting Datasets 1, 2, 3, and 4 generation pipeline...")
    runner = CoSimulationRunner()
    cla_estimator = ClusterLoadAllocationEstimator()
    time_cla_estimator = TimeAdjustedCLAEstimator()

    rows_1 = []

    # =========================================================================
    # --- A. DATASET 1 GENERATION (Single 5-minute steady-state experiment) ---
    # =========================================================================
    print("INFO: Starting Dataset 1 generation...")
    print("INFO: Initializing OpenDSS instance for Dataset 1 generation...")
    runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

    sim_res_d1 = runner.run_simulation(
        use_baseline_transformers=True,
        is_steady_state_run=True,
        scenario_id="steady_5min_run",
        seed=42,
        reinitialize_plant=False
    )

    registry = runner.plant_data["registry"] if runner.plant_data else None
    registered_units = registry.get_registered_consumers() if registry else []
    latent_map = {c.bus_id: c for c in registry.get_latent_consumers()} if registry else {}

    for f_id in [1, 2, 3]:
        feeder_units = [u for u in registered_units if u.feeder_id == f"feeder_{f_id}"]
        if not feeder_units:
            continue

        m_key = f"trans{f_id}_lv_boundary_consumer_unit"
        meas = sim_res_d1.steady_state_measurements.get(m_key, {})

        sampled_units = [u for u in feeder_units if u.is_metered]
        unsampled_units = [u for u in feeder_units if not u.is_metered]

        gt_sampled_energy_kwh = round(
            sum(
                float(sim_res_d1.steady_state_measurements.get(u.consumer_id, {}).get("energy_kwh", 0.0))
                for u in sampled_units
            ),
            4
        )

        gt_unsampled_true_kwh = round(
            sum(
                float(sim_res_d1.steady_state_measurements.get(u.consumer_id, {}).get("energy_kwh", 0.0))
                for u in unsampled_units
            ),
            4
        )

        gt_latent_energy_kwh = round(
            sum(
                float(sim_res_d1.steady_state_measurements.get(latent_map[u.bus_id].consumer_id, {}).get("energy_kwh", 0.0))
                for u in feeder_units if u.bus_id in latent_map
            ),
            4
        )

        # Dynamic physics-based transformer losses and line losses (for 5-min run, dt = 5/60 hours)
        transformer_loss_kw = float(meas.get("transformer_losses", 0.0))
        transformer_loss_kwh = round(transformer_loss_kw * (5.0 / 60.0), 4)

        feeder_line_loss_kw = float(meas.get("feeder_line_losses", 0.0))
        consumer_line_loss_kw = sum(
            float(sim_res_d1.steady_state_measurements.get(u.consumer_id, {}).get("line_losses", 0.0))
            for u in feeder_units
        )
        line_loss_kwh = round((feeder_line_loss_kw + consumer_line_loss_kw) * (5.0 / 60.0), 4)

        gt_tech_loss_kwh = round(transformer_loss_kwh + line_loss_kwh, 4)

        feeder_supply_energy_kwh = round(
            gt_sampled_energy_kwh + gt_unsampled_true_kwh + gt_latent_energy_kwh + gt_tech_loss_kwh,
            4
        )

        # Estimate energy for unsampled units
        unsampled_premises = [
            ConsumerLoadPremises(
                consumer_id=u.consumer_id,
                class_id=u.assigned_load_class or "residential",
                is_sampled=False,
                connected_load_kw=float(len(u.loads) * 5.0)
            )
            for u in unsampled_units
        ]

        metered_premises = [
            ConsumerLoadPremises(
                consumer_id=u.consumer_id,
                class_id=u.assigned_load_class or "residential",
                is_sampled=True,
                connected_load_kw=float(len(u.loads) * 5.0)
            )
            for u in sampled_units
        ]

        metered_consumer_energies = {
            u.consumer_id: float(sim_res_d1.steady_state_measurements.get(u.consumer_id, {}).get("energy_kwh", 0.0))
            for u in sampled_units
        }

        is_valid_cla = cla_estimator.validation_function(
            feeder_supply_energy_kwh=feeder_supply_energy_kwh,
            sampled_consumer_energy_kwh=gt_sampled_energy_kwh,
            estimated_technical_loss_kwh=gt_tech_loss_kwh
        )

        cla_res = cla_estimator.estimate(
            feeder_supply_energy_kwh=feeder_supply_energy_kwh,
            sampled_consumer_energy_kwh=gt_sampled_energy_kwh,
            estimated_technical_loss_kwh=gt_tech_loss_kwh,
            unsampled_premises=unsampled_premises
        ) if (is_valid_cla and unsampled_premises) else None

        time_cla_res = time_cla_estimator.estimate(
            feeder_supply_energy_kwh=feeder_supply_energy_kwh,
            sampled_consumer_energy_kwh=gt_sampled_energy_kwh,
            estimated_technical_loss_kwh=gt_tech_loss_kwh,
            unsampled_premises=unsampled_premises,
            metered_premises=metered_premises,
            metered_consumer_energies=metered_consumer_energies
        ) if (is_valid_cla and unsampled_premises) else None

        weights_map = cla_estimator.weighting_function(unsampled_premises) if unsampled_premises else {}

        for u in feeder_units:
            is_metered = u.is_metered

            unit_meas = sim_res_d1.steady_state_measurements.get(u.consumer_id, {})
            unit_dss_energy = float(unit_meas.get("energy_kwh", 0.0))

            meas_energy = round(unit_dss_energy, 4) if is_metered else np.nan
            cla_est = round(float(cla_res.allocated_unsampled_consumer_energy.get(u.consumer_id, 0.0)), 4) if (not is_metered and cla_res and u.consumer_id in cla_res.allocated_unsampled_consumer_energy) else np.nan
            time_cla_est = round(float(time_cla_res.allocated_unsampled_consumer_energy.get(u.consumer_id, 0.0)), 4) if (not is_metered and time_cla_res and u.consumer_id in time_cla_res.allocated_unsampled_consumer_energy) else np.nan

            unit_weight = round(float(weights_map.get(u.consumer_id, np.nan)), 6) if not is_metered else np.nan

            assigned_class = u.assigned_load_class if u.assigned_load_class else "residential"
            consumer_type_label = f"{assigned_class}_{'metered' if is_metered else 'unmetered'}"

            c_line_loss = round(float(unit_meas.get("line_losses", 0.0)), 4)

            # Registered consumer unit (consumer_type includes assigned class and status type)
            rows_1.append({
                "gt_consumer_unit_id": u.consumer_id,
                "consumer_type": consumer_type_label,
                "consumer_unit_source": json.dumps({"bus": u.bus_id, "feeder": u.feeder_id}),
                "consumer_unit_loads": json.dumps([{"load_id": ld.load_id, "circuit_id": ld.circuit_id, "load_type": ld.load_type} for ld in u.loads]),
                "assigned_weight": unit_weight,
                "gt_consumed_energy_kwh": round(unit_dss_energy, 4),
                "consumer_line_losses": c_line_loss,
                "measured_energy_kwh": meas_energy,
                "cla_estimates": cla_est,
                "time_adjusted_cla_estimates": time_cla_est
            })

            # Latent / unknown consumer unit at same bus if present
            latent_u = latent_map.get(u.bus_id)
            if latent_u:
                latent_meas = sim_res_d1.steady_state_measurements.get(latent_u.consumer_id, {})
                latent_line_loss = round(float(latent_meas.get("line_losses", 0.0)), 4)
                rows_1.append({
                    "gt_consumer_unit_id": latent_u.consumer_id,
                    "consumer_type": "latent",
                    "consumer_unit_source": json.dumps({"bus": latent_u.bus_id, "feeder": latent_u.feeder_id}),
                    "consumer_unit_loads": json.dumps([{"load_id": ld.load_id, "circuit_id": ld.circuit_id, "load_type": ld.load_type} for ld in latent_u.loads]),
                    "assigned_weight": np.nan,
                    "gt_consumed_energy_kwh": np.nan,
                    "consumer_line_losses": latent_line_loss,
                    "measured_energy_kwh": np.nan,
                    "cla_estimates": np.nan,
                    "time_adjusted_cla_estimates": np.nan
                })

    print("INFO: Completed Dataset 1 generation.")

    all_108_pairs = get_all_108_coevents(target_line="feeder1_head")

    # =========================================================================
    # --- B. DATASET 2 GENERATION (108 Unique Co-Events Parallel Batches) ---
    # =========================================================================
    rows_2 = process_dataset_coevents_in_batches(all_108_pairs, use_baseline_transformers=True, is_dataset_3=False, dataset_name="Dataset 2", batch_size=6)

    # =========================================================================
    # --- C. DATASET 3 GENERATION (108 Unique Co-Events Time Shift Parallel) ---
    # =========================================================================
    rows_3 = process_dataset_coevents_in_batches(all_108_pairs, use_baseline_transformers=True, is_dataset_3=True, dataset_name="Dataset 3", batch_size=6)

    # =========================================================================
    # --- D. DATASET 4 GENERATION (108 Unique Co-Events Transformer Spec Parallel) ---
    # =========================================================================
    rows_4 = process_dataset_coevents_in_batches(all_108_pairs, use_baseline_transformers=False, is_dataset_3=False, dataset_name="Dataset 4", batch_size=6)

    df_1 = pd.DataFrame(rows_1)
    df_2 = pd.DataFrame(rows_2)
    df_3 = pd.DataFrame(rows_3)
    df_4 = pd.DataFrame(rows_4)

    if write_to_disk:
        dir_path = Path("src/simulation")
        dir_path.mkdir(parents=True, exist_ok=True)
        df_1.to_csv(dir_path / "dataset_1.csv", index=False)
        df_2.to_csv(dir_path / "dataset_2.csv", index=False)
        df_3.to_csv(dir_path / "dataset_3.csv", index=False)
        df_4.to_csv(dir_path / "dataset_4.csv", index=False)

        dumps_dir = dir_path / "time_series_dumps"
        dumps_dir.mkdir(parents=True, exist_ok=True)
        for ds_name, df_ds in [("dataset_2", df_2), ("dataset_3", df_3), ("dataset_4", df_4)]:
            dumps = []
            for idx, row in df_ds.iterrows():
                entry = {}
                for col in df_ds.columns:
                    if col.startswith("obs_"):
                        val = row[col]
                        if isinstance(val, str):
                            try:
                                entry[col] = json.loads(val)
                            except Exception:
                                entry[col] = val
                        else:
                            entry[col] = val
                dumps.append(entry)
            with open(dumps_dir / f"{ds_name}_time_series_dumps.json", "w") as f:
                json.dump(dumps, f, indent=1)

        print(f"INFO: Successfully written datasets to {dir_path / 'dataset_1.csv'}, {dir_path / 'dataset_2.csv'}, {dir_path / 'dataset_3.csv'}, and {dir_path / 'dataset_4.csv'}")

    return df_1, df_2, df_3, df_4


if __name__ == "__main__":
    generate_experiments_dataset(write_to_disk=True)
