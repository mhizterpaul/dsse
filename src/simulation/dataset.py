import os
import sys
import csv
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

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


def generate_experiments_dataset(write_to_disk: bool = True):
    """
    Orchestrates dataset generation for Dataset 1 (Cluster Load Allocation energy estimation),
    Dataset 2 (108 unique co-events observability), Dataset 3 (108 unique co-events time shift),
    and Dataset 4 (108 unique co-events transformer spec effect) using CoSimulationRunner functions ONLY.
    """
    print("INFO: Generating Datasets 1, 2, 3, and 4...")
    runner = CoSimulationRunner()
    cla_estimator = ClusterLoadAllocationEstimator()
    time_cla_estimator = TimeAdjustedCLAEstimator()

    rows_1 = []
    rows_2 = []
    rows_3 = []
    rows_4 = []

    # =========================================================================
    # --- A. DATASET 1 GENERATION (Single 5-minute steady-state experiment) ---
    # =========================================================================
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

        num_sampled = int(len(feeder_units) * 0.36)
        sampled_units = feeder_units[:num_sampled]
        unsampled_units = feeder_units[num_sampled:]

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

        num_sampled = int(len(feeder_units) * 0.36)

        for u_idx, u in enumerate(feeder_units):
            is_metered = u_idx < num_sampled

            unit_meas = sim_res_d1.steady_state_measurements.get(u.consumer_id, {})
            unit_dss_energy = float(unit_meas.get("energy_kwh", 0.0))

            meas_energy = round(unit_dss_energy, 4) if is_metered else ""
            cla_est = round(float(cla_res.allocated_unsampled_consumer_energy.get(u.consumer_id, cla_res.estimated_unsampled_energy_kwh / len(unsampled_premises))), 4) if (not is_metered and cla_res and unsampled_premises) else ""
            time_cla_est = round(float(time_cla_res.allocated_unsampled_consumer_energy.get(u.consumer_id, time_cla_res.estimated_unsampled_energy_kwh / len(unsampled_premises))), 4) if (not is_metered and time_cla_res and unsampled_premises) else ""

            unit_weight = round(float(weights_map.get(u.consumer_id, 1.0)), 4) if not is_metered else ""

            # Known / registered consumer unit
            rows_1.append({
                "gt_consumer_unit_id": u.consumer_id,
                "consumer_type": "known",
                "consumer_unit_source": json.dumps({"bus": u.bus_id, "feeder": u.feeder_id}),
                "consumer_unit_loads": json.dumps([{"load_id": ld.load_id, "circuit_id": ld.circuit_id, "load_type": ld.load_type} for ld in u.loads]),
                "assigned_weight": unit_weight,
                "gt_consumed_energy_kwh": round(unit_dss_energy, 4),
                "consumer_line_losses": unit_meas.get("line_losses", 0.15),
                "measured_energy_kwh": meas_energy,
                "cla_estimates": cla_est,
                "time_adjusted_cla_estimates": time_cla_est
            })

            # Latent / unknown consumer unit at same bus if present
            latent_u = latent_map.get(u.bus_id)
            if latent_u:
                rows_1.append({
                    "gt_consumer_unit_id": latent_u.consumer_id,
                    "consumer_type": "unknown",
                    "consumer_unit_source": json.dumps({"bus": latent_u.bus_id, "feeder": latent_u.feeder_id}),
                    "consumer_unit_loads": json.dumps([{"load_id": ld.load_id, "circuit_id": ld.circuit_id, "load_type": ld.load_type} for ld in latent_u.loads]),
                    "assigned_weight": "",
                    "gt_consumed_energy_kwh": "",
                    "consumer_line_losses": 0.10,
                    "measured_energy_kwh": "",
                    "cla_estimates": "",
                    "time_adjusted_cla_estimates": ""
                })

    all_108_pairs = get_all_108_coevents(target_line="feeder1_head")

    def extract_load_source(co_ev):
        sources = []
        for ev in [co_ev.event_1, co_ev.event_2]:
            if getattr(ev, "event_class", "") == "equipment_switch":
                target = getattr(ev, "target", "feeder1_head")
                sources.append({"bus": target, "line": f"line_{target}"})
        return json.dumps(sources) if sources else ""

    def process_coevent_simulation(co_ev, f_id, use_baseline_transformers: bool = True):
        ev1, ev2 = co_ev.event_1, co_ev.event_2
        m_id = f"trans{f_id}_lv_boundary_consumer_unit"

        # 1. Simulate combined co-event (load transients powered under stipulated network conditions)
        sim_co = runner.run_simulation(
            events=[co_ev],
            use_baseline_transformers=use_baseline_transformers,
            include_load_event=True,
            include_fault_event=True,
            scenario_id=f"coev_{f_id}_{getattr(ev1, 'event_type', '')}_{getattr(ev2, 'event_type', '')}",
            seed=42,
            reinitialize_plant=False
        )
        unit_co = sim_co.processed_consumer_units.get(m_id, {})
        v_co = np.array(unit_co.get("raw_voltage", np.zeros((1000, 3))))
        i_co = np.array(unit_co.get("raw_current", np.zeros((1000, 3))))

        # 2. Simulate constituent event 1
        sim_1 = runner.run_simulation(
            events=[ev1],
            use_baseline_transformers=use_baseline_transformers,
            include_load_event=True,
            include_fault_event=(getattr(ev1, "event_class", "") == "line_fault"),
            scenario_id=f"ev1_{f_id}_{getattr(ev1, 'event_type', '')}",
            seed=42,
            reinitialize_plant=False
        )
        unit_1 = sim_1.processed_consumer_units.get(m_id, {})
        v1_sig = np.array(unit_1.get("raw_voltage", np.zeros((1000, 3))))
        i1_sig = np.array(unit_1.get("raw_current", np.zeros((1000, 3))))

        # 3. Simulate constituent event 2
        sim_2 = runner.run_simulation(
            events=[ev2],
            use_baseline_transformers=use_baseline_transformers,
            include_load_event=True,
            include_fault_event=(getattr(ev2, "event_class", "") == "line_fault"),
            scenario_id=f"ev2_{f_id}_{getattr(ev2, 'event_type', '')}",
            seed=42,
            reinitialize_plant=False
        )
        unit_2 = sim_2.processed_consumer_units.get(m_id, {})
        v2_sig = np.array(unit_2.get("raw_voltage", np.zeros((1000, 3))))
        i2_sig = np.array(unit_2.get("raw_current", np.zeros((1000, 3))))

        t_s = sim_co.time_s if len(sim_co.time_s) > 0 else np.linspace(0.0, 0.1, 1000)

        # High-pass filter low-frequency fundamentals
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

        return t_s, v_co, i_co, v1_hp, i1_hp, v2_hp, i2_hp, v_comp, i_comp, res_v, res_i, v_mag, i_mag

    # =========================================================================
    # --- B. DATASET 2 GENERATION (108 Unique Co-Events) ---
    # =========================================================================
    print("INFO: Initializing OpenDSS instance for Dataset 2 generation loop (108 unique co-events)...")
    runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

    for p_idx, (pair_cat, co_ev) in enumerate(all_108_pairs):
        ev1, ev2 = co_ev.event_1, co_ev.event_2
        f_id = (p_idx % 3) + 1

        t_s, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag = process_coevent_simulation(
            co_ev, f_id, use_baseline_transformers=True
        )

        rows_2.append({
            "load_source": extract_load_source(co_ev),
            "fault_info": extract_fault_info(co_ev),
            "gt_event_1_equipment_type": getattr(ev1, "equipment_type", ""),
            "gt_event_1_fault_type": getattr(ev1, "fault_type", ""),
            "gt_event_1_start_timestamp_s": float(ev1.start_time_s),
            "gt_event_2_equipment_type": getattr(ev2, "equipment_type", ""),
            "gt_event_2_fault_type": getattr(ev2, "fault_type", ""),
            "gt_event_2_start_timestamp_s": float(ev2.start_time_s),
            "gt_time_offset_s": 0.0,
            "obs_coevent_time": json.dumps(t_s.tolist()),
            "obs_coevent_v": json.dumps([v_co[:, 0].tolist(), v_co[:, 1].tolist(), v_co[:, 2].tolist()]),
            "obs_coevent_i": json.dumps([i_co[:, 0].tolist(), i_co[:, 1].tolist(), i_co[:, 2].tolist()]),

            "obs_single_event_1_v_phase_a": json.dumps(v1_sig[:, 0].tolist()),
            "obs_single_event_1_v_phase_b": json.dumps(v1_sig[:, 1].tolist()),
            "obs_single_event_1_v_phase_c": json.dumps(v1_sig[:, 2].tolist()),
            "obs_single_event_1_i_phase_a": json.dumps(i1_sig[:, 0].tolist()),
            "obs_single_event_1_i_phase_b": json.dumps(i1_sig[:, 1].tolist()),
            "obs_single_event_1_i_phase_c": json.dumps(i1_sig[:, 2].tolist()),

            "obs_single_event_2_v_phase_a": json.dumps(v2_sig[:, 0].tolist()),
            "obs_single_event_2_v_phase_b": json.dumps(v2_sig[:, 1].tolist()),
            "obs_single_event_2_v_phase_c": json.dumps(v2_sig[:, 2].tolist()),
            "obs_single_event_2_i_phase_a": json.dumps(i2_sig[:, 0].tolist()),
            "obs_single_event_2_i_phase_b": json.dumps(i2_sig[:, 1].tolist()),
            "obs_single_event_2_i_phase_c": json.dumps(i2_sig[:, 2].tolist()),

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
        })

    # =========================================================================
    # --- C. DATASET 3 GENERATION (108 Unique Co-Events with Time Shift) ---
    # =========================================================================
    print("INFO: Initializing OpenDSS instance for Dataset 3 generation loop (108 unique co-events)...")
    runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

    for p_idx, (pair_cat, co_ev) in enumerate(all_108_pairs):
        ev1, ev2 = co_ev.event_1, co_ev.event_2
        time_offset = 0.01 if p_idx % 2 == 1 else 0.0  # alternate simultaneous vs shifted
        shifted_co_ev = co_ev.with_time_shift(time_offset) if time_offset > 0 else co_ev
        f_id = (p_idx % 3) + 1

        t_s, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag = process_coevent_simulation(
            shifted_co_ev, f_id, use_baseline_transformers=True
        )

        rows_3.append({
            "load_source": extract_load_source(shifted_co_ev),
            "fault_info": extract_fault_info(shifted_co_ev),
            "gt_event_1_equipment_type": getattr(ev1, "equipment_type", ""),
            "gt_event_1_fault_type": getattr(ev1, "fault_type", ""),
            "gt_event_1_start_timestamp_s": float(ev1.start_time_s),
            "gt_event_2_equipment_type": getattr(ev2, "equipment_type", ""),
            "gt_event_2_fault_type": getattr(ev2, "fault_type", ""),
            "gt_event_2_start_timestamp_s": float(shifted_co_ev.event_2.start_time_s),
            "gt_time_offset_s": float(time_offset),
            "obs_coevent_time": json.dumps(t_s.tolist()),
            "obs_coevent_v": json.dumps([v_co[:, 0].tolist(), v_co[:, 1].tolist(), v_co[:, 2].tolist()]),
            "obs_coevent_i": json.dumps([i_co[:, 0].tolist(), i_co[:, 1].tolist(), i_co[:, 2].tolist()]),

            "obs_single_event_1_v_phase_a": json.dumps(v1_sig[:, 0].tolist()),
            "obs_single_event_1_v_phase_b": json.dumps(v1_sig[:, 1].tolist()),
            "obs_single_event_1_v_phase_c": json.dumps(v1_sig[:, 2].tolist()),
            "obs_single_event_1_i_phase_a": json.dumps(i1_sig[:, 0].tolist()),
            "obs_single_event_1_i_phase_b": json.dumps(i1_sig[:, 1].tolist()),
            "obs_single_event_1_i_phase_c": json.dumps(i1_sig[:, 2].tolist()),

            "obs_single_event_2_v_phase_a": json.dumps(v2_sig[:, 0].tolist()),
            "obs_single_event_2_v_phase_b": json.dumps(v2_sig[:, 1].tolist()),
            "obs_single_event_2_v_phase_c": json.dumps(v2_sig[:, 2].tolist()),
            "obs_single_event_2_i_phase_a": json.dumps(i2_sig[:, 0].tolist()),
            "obs_single_event_2_i_phase_b": json.dumps(i2_sig[:, 1].tolist()),
            "obs_single_event_2_i_phase_c": json.dumps(i2_sig[:, 2].tolist()),

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
        })

    # =========================================================================
    # --- D. DATASET 4 GENERATION (108 Unique Co-Events with Transformer Spec Effect) ---
    # =========================================================================
    print("INFO: Initializing OpenDSS instance for Dataset 4 generation loop (108 unique co-events)...")
    runner.initialize_plant_session(use_baseline_transformers=False, seed=42)

    for p_idx, (pair_cat, co_ev) in enumerate(all_108_pairs):
        ev1, ev2 = co_ev.event_1, co_ev.event_2
        f_id = (p_idx % 3) + 1

        t_s, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag = process_coevent_simulation(
            co_ev, f_id, use_baseline_transformers=False
        )

        rows_4.append({
            "gt_feeder_id": f"feeder_{f_id}",
            "load_source": extract_load_source(co_ev),
            "fault_info": extract_fault_info(co_ev),
            "gt_event_1_equipment_type": getattr(ev1, "equipment_type", ""),
            "gt_event_1_fault_type": getattr(ev1, "fault_type", ""),
            "gt_event_1_start_timestamp_s": float(ev1.start_time_s),
            "gt_event_2_equipment_type": getattr(ev2, "equipment_type", ""),
            "gt_event_2_fault_type": getattr(ev2, "fault_type", ""),
            "gt_event_2_start_timestamp_s": float(ev2.start_time_s),
            "gt_time_offset_s": 0.0,
            "obs_coevent_time": json.dumps(t_s.tolist()),
            "obs_coevent_v": json.dumps([v_co[:, 0].tolist(), v_co[:, 1].tolist(), v_co[:, 2].tolist()]),
            "obs_coevent_i": json.dumps([i_co[:, 0].tolist(), i_co[:, 1].tolist(), i_co[:, 2].tolist()]),

            "obs_single_event_1_v_phase_a": json.dumps(v1_sig[:, 0].tolist()),
            "obs_single_event_1_v_phase_b": json.dumps(v1_sig[:, 1].tolist()),
            "obs_single_event_1_v_phase_c": json.dumps(v1_sig[:, 2].tolist()),
            "obs_single_event_1_i_phase_a": json.dumps(i1_sig[:, 0].tolist()),
            "obs_single_event_1_i_phase_b": json.dumps(i1_sig[:, 1].tolist()),
            "obs_single_event_1_i_phase_c": json.dumps(i1_sig[:, 2].tolist()),

            "obs_single_event_2_v_phase_a": json.dumps(v2_sig[:, 0].tolist()),
            "obs_single_event_2_v_phase_b": json.dumps(v2_sig[:, 1].tolist()),
            "obs_single_event_2_v_phase_c": json.dumps(v2_sig[:, 2].tolist()),
            "obs_single_event_2_i_phase_a": json.dumps(i2_sig[:, 0].tolist()),
            "obs_single_event_2_i_phase_b": json.dumps(i2_sig[:, 1].tolist()),
            "obs_single_event_2_i_phase_c": json.dumps(i2_sig[:, 2].tolist()),

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
        })

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
