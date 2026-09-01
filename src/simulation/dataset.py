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
from src.estimator.cla_estimator import ClusterLoadAllocationEstimator
from src.estimator.time_adjusted_cla_estimator import TimeAdjustedCLAEstimator
from src.loads import get_equipment_model



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

    sim_res_d1 = runner.run_steady_state_simulation(
        use_baseline_transformers=True,
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

        duration_hours = 5.0 / 60.0  # Total time network energized (5 minutes = 300 s / 3600 h)

        # Query consumer unit energies directly from OpenDSS load powers
        consumer_energies = {}
        consumer_loss_kws = {}
        for u in feeder_units:
            unit_p_kw = 0.0
            for ld in u.loads:
                if runner.dss.Circuit.SetActiveElement(f"load.{ld.load_id}"):
                    powers = runner.dss.CktElement.Powers()
                    if len(powers) >= 2:
                        unit_p_kw += sum(powers[0::2])
            unit_energy_kwh = unit_p_kw * duration_hours
            consumer_energies[u.consumer_id] = unit_energy_kwh

            if u.bus_id in latent_map:
                lat_u = latent_map[u.bus_id]
                lat_p_kw = 0.0
                for ld in lat_u.loads:
                    if runner.dss.Circuit.SetActiveElement(f"load.{ld.load_id}"):
                        powers = runner.dss.CktElement.Powers()
                        if len(powers) >= 2:
                            lat_p_kw += sum(powers[0::2])
                consumer_energies[lat_u.consumer_id] = lat_p_kw * duration_hours

        sampled_units = [u for u in feeder_units if u.is_metered]
        unsampled_units = [u for u in feeder_units if not u.is_metered]

        gt_sampled_energy_kwh = round(
            sum(consumer_energies.get(u.consumer_id) for u in sampled_units),
            4
        )

        gt_unsampled_true_kwh = round(
            sum(consumer_energies.get(u.consumer_id) for u in unsampled_units),
            4
        )

        gt_latent_energy_kwh = round(
            sum(consumer_energies.get(latent_map[u.bus_id].consumer_id) for u in feeder_units if u.bus_id in latent_map),
            4
        )

        # Dynamic physics-based transformer losses and line losses from OpenDSS
        tx_name = f"Transformer.trans{f_id}"
        feeder_name = f"Line.feeder{f_id}"

        transformer_loss_kw = 0.0
        if runner.dss.Circuit.SetActiveElement(tx_name):
            tx_losses = runner.dss.CktElement.Losses()
            if len(tx_losses) >= 1:
                transformer_loss_kw = abs(tx_losses[0]) / 1000.0
        transformer_loss_kwh = round(transformer_loss_kw * duration_hours, 4)

        feeder_line_loss_kw = 0.0
        if runner.dss.Circuit.SetActiveElement(feeder_name):
            f_losses = runner.dss.CktElement.Losses()
            if len(f_losses) >= 1:
                feeder_line_loss_kw = abs(f_losses[0]) / 1000.0

        # Calculate consumer line loss kW using actual OpenDSS bus voltages and currents
        consumer_line_loss_kw = 0.0
        for u in feeder_units:
            try:
                # Set active bus first per OpenDSS API requirements
                bus_idx = runner.dss.Circuit.SetActiveBus(u.bus_id)
                v_vec = np.array(runner.dss.Bus.VMagAngle())
                if len(v_vec) < 2 or np.mean(v_vec[0::2]) <= 0:
                    all_buses = runner.dss.Circuit.AllBusNames()
                    raise ValueError(
                        f"Missing or non-positive voltage magnitude for bus '{u.bus_id}' (v_vec: {v_vec}). "
                        f"Bus index: {bus_idx}. Available circuit buses count: {len(all_buses)}"
                    )
            except Exception as e:
                print(f"ERROR: Exception occurred while querying bus '{u.bus_id}' for consumer unit '{u.consumer_id}': {e}\n{traceback.format_exc()}")
                raise

            v_ln = float(np.mean(v_vec[0::2]))
            v_ll = v_ln * (3.0 ** 0.5)
            i_total_line = 0.0
            for ld in u.loads:
                eq = get_equipment_model(ld.load_type)
                p_w = eq.rated_power_kw * 1000.0
                if eq.power_factor <= 0:
                    raise ValueError(f"Equipment '{ld.load_type}' has non-positive power factor: {eq.power_factor}")
                i_ld = p_w / ((3.0 ** 0.5) * v_ll * eq.power_factor)
                i_total_line += i_ld
            p_loss_kw = 3.0 * (i_total_line ** 2) * u.service_line_resistance_ohm / 1000.0
            consumer_line_loss_kw += p_loss_kw

        line_loss_kwh = round((feeder_line_loss_kw + consumer_line_loss_kw) * duration_hours, 4)

        gt_tech_loss_kwh = round(transformer_loss_kwh + line_loss_kwh, 4)

        feeder_supply_energy_kwh = round(
            gt_sampled_energy_kwh + gt_unsampled_true_kwh + gt_latent_energy_kwh + gt_tech_loss_kwh,
            4
        )

        

        metered_consumer_energies = {
            u.consumer_id: consumer_energies[u.consumer_id]
            for u in sampled_units
        }

        # Store in steady state measurements
        sim_res_d1.steady_state_measurements = {
            u.consumer_id: {"energy_kwh": consumer_energies[u.consumer_id]}
            for u in feeder_units
        }

        cla_res = cla_estimator.estimate(
            feeder_supply_energy_kwh=feeder_supply_energy_kwh,
            sampled_consumer_energy_kwh=gt_sampled_energy_kwh,
            technical_loss_kwh=gt_tech_loss_kwh,
            registry=registry
        ) 

        time_cla_res = time_cla_estimator.estimate(
            feeder_supply_energy_kwh=feeder_supply_energy_kwh,
            technical_loss_kwh=gt_tech_loss_kwh,
            metered_consumer_energies=metered_consumer_energies,
            registry=registry
        ) 

        unmetered_units = [u for u in feeder_units if not u.is_metered]
        weights_map = cla_estimator.weighting_function(unmetered_units)

        duration_hours = 5.0 / 60.0  # Total time network energized (5 minutes = 300 s / 3600 h)

        for u in feeder_units:
            is_metered = u.is_metered

            try:
                bus_idx = runner.dss.Circuit.SetActiveBus(u.bus_id)
                v_vec = np.array(runner.dss.Bus.VMagAngle())
                if len(v_vec) < 2 or np.mean(v_vec[0::2]) <= 0:
                    raise ValueError(f"Missing or non-positive voltage magnitude for bus '{u.bus_id}' (v_vec: {v_vec}, bus_idx: {bus_idx})")
            except Exception as e:
                print(f"ERROR: Exception occurred while querying bus '{u.bus_id}' for consumer unit '{u.consumer_id}': {e}\n{traceback.format_exc()}")
                raise

            v_ln = float(np.mean(v_vec[0::2]))
            v_ll = v_ln * (3.0 ** 0.5)

            # Compute gt_consumed_energy_kwh = 3 * |Z_circuits| * I_line^2 * pf_eff * t
            i_total_line = 0.0
            pf_eff = 0.0
            num_loads = len(u.loads)
            if num_loads == 0:
                raise ValueError(f"Consumer unit '{u.consumer_id}' has no connected load circuits")

            for ld in u.loads:
                eq = get_equipment_model(ld.load_type)
                p_w = eq.rated_power_kw * 1000.0
                if eq.power_factor <= 0:
                    raise ValueError(f"Equipment '{ld.load_type}' has invalid power factor: {eq.power_factor}")
                i_ld = p_w / ((3.0 ** 0.5) * v_ll * eq.power_factor)
                i_total_line += i_ld
                pf_eff += eq.power_factor

            pf_eff = pf_eff / num_loads
            if i_total_line <= 0:
                raise ValueError(f"Total current for consumer unit '{u.consumer_id}' is non-positive: {i_total_line}")
            z_mag = v_ln / i_total_line
            unit_consumed_energy_kwh = (3.0 * z_mag * (i_total_line ** 2) * pf_eff * duration_hours) / 1000.0

            # Calculate service drop line loss directly from consumer unit service line resistance
            r_drop = u.service_line_resistance_ohm
            p_loss_kw = 3.0 * (i_total_line ** 2) * r_drop / 1000.0
            c_line_loss_kwh = round(float(p_loss_kw * duration_hours), 6)

            cla_est = round(float(cla_res.allocated_unsampled_consumer_energy.get(u.consumer_id)), 4) if (not is_metered and cla_res and u.consumer_id in cla_res.allocated_unsampled_consumer_energy) else np.nan
            time_cla_est = round(float(time_cla_res.allocated_unsampled_consumer_energy.get(u.consumer_id)), 4) if (not is_metered and time_cla_res and u.consumer_id in time_cla_res.allocated_unsampled_consumer_energy) else np.nan

            unit_weight = round(float(weights_map.get(u.consumer_id)), 6) if not is_metered else np.nan

            assigned_class = u.assigned_load_class 
            consumer_type_label = f"{assigned_class}_{'metered' if is_metered else 'unmetered'}"

            # Registered consumer unit (consumer_type includes assigned class and status type)
            rows_1.append({
                "gt_consumer_unit_id": u.consumer_id,
                "consumer_type": consumer_type_label,
                "consumer_unit_source": json.dumps({"bus": u.bus_id, "feeder": u.feeder_id}),
                "consumer_unit_loads": json.dumps([{"load_id": ld.load_id, "circuit_id": ld.circuit_id, "load_type": ld.load_type} for ld in u.loads]),
                "assigned_weight": unit_weight,
                "gt_consumed_energy_kwh": round(unit_consumed_energy_kwh, 4),
                "consumer_line_losses": c_line_loss_kwh,
                "cla_estimates": cla_est,
                "time_adjusted_cla_estimates": time_cla_est
            })

            # Latent / unknown consumer unit at same bus if present
            latent_u = latent_map.get(u.bus_id)
            if latent_u:
                i_latent_total = 0.0
                if len(latent_u.loads) == 0:
                    raise ValueError(f"Latent consumer unit '{latent_u.consumer_id}' has no connected load circuits")

                for ld in latent_u.loads:
                    eq = get_equipment_model(ld.load_type)
                    p_w = eq.rated_power_kw * 1000.0
                    if eq.power_factor <= 0:
                        raise ValueError(f"Latent equipment '{ld.load_type}' has invalid power factor: {eq.power_factor}")
                    i_ld = p_w / ((3.0 ** 0.5) * v_ll * eq.power_factor)
                    i_latent_total += i_ld

                r_latent_drop = latent_u.service_line_resistance_ohm
                p_latent_loss_kw = 3.0 * (i_latent_total ** 2) * r_latent_drop / 1000.0
                latent_line_loss_kwh = round(float(p_latent_loss_kw * duration_hours), 6)

                rows_1.append({
                    "gt_consumer_unit_id": latent_u.consumer_id,
                    "consumer_type": "latent",
                    "consumer_unit_source": json.dumps({"bus": latent_u.bus_id, "feeder": latent_u.feeder_id}),
                    "consumer_unit_loads": json.dumps([{"load_id": ld.load_id, "circuit_id": ld.circuit_id, "load_type": ld.load_type} for ld in latent_u.loads]),
                    "assigned_weight": np.nan,
                    "gt_consumed_energy_kwh": np.nan,
                    "consumer_line_losses": latent_line_loss_kwh,
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
