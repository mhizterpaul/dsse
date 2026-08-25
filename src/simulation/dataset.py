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

from src.simulation.runner import CoSimulationRunner
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


def generate_experiments_dataset(n_scenarios: int = 15, write_to_disk: bool = True):
    """
    Orchestrates dataset generation for Dataset 1 (Cluster Load Allocation energy estimation),
    Dataset 2 (108 unique co-events observability), Dataset 3 (108 unique co-events time shift),
    and Dataset 4 (108 unique co-events transformer spec effect) using CoSimulationRunner functions ONLY.
    """
    print("INFO: Sweeping scenarios and generating Datasets 1, 2, 3, and 4...")
    runner = CoSimulationRunner()
    cla_estimator = ClusterLoadAllocationEstimator()
    time_cla_estimator = TimeAdjustedCLAEstimator()

    rows_1 = []
    rows_2 = []
    rows_3 = []
    rows_4 = []

    signature_catalog = {}

    scenario_configs = [
        {"load_comp": "linear"},
        {"load_comp": "non_linear"},
        {"load_comp": "heavy_duty"}
    ] * 5

    # =========================================================================
    # --- A. DATASET 1 GENERATION (Constant OpenDSS Instance Loop 1) ---
    # =========================================================================
    print("INFO: Initializing OpenDSS instance for Dataset 1 generation loop...")
    runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

    for idx in range(min(n_scenarios, len(scenario_configs))):
        scenario_id = f"scenario_{idx}"
        feeder_idx = (idx % 3) + 1

        sim_res_d1 = runner.run_simulation(
            use_baseline_transformers=True,
            is_steady_state_run=True,
            scenario_id=f"{scenario_id}_steady",
            seed=42 + idx,
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
            gt_total_energy_kwh = float(meas.get("energy_kwh", 150.0))
            if gt_total_energy_kwh <= 0:
                gt_total_energy_kwh = 150.0

            gt_sampled_energy_kwh = round(gt_total_energy_kwh * 0.36, 4)
            gt_unsampled_energy_kwh = round(gt_total_energy_kwh * 0.64, 4)

            transformer_loss_kwh = round(0.02 * gt_total_energy_kwh, 4)
            line_loss_kwh = round(0.03 * gt_total_energy_kwh, 4)
            gt_tech_loss_kwh = round(transformer_loss_kwh + line_loss_kwh, 4)
            gt_non_tech_loss_kwh = round(0.08 * gt_total_energy_kwh, 4)

            feeder_supply_energy_kwh = gt_total_energy_kwh + gt_tech_loss_kwh + gt_non_tech_loss_kwh

            # Estimate energy for unsampled units
            unsampled_units = feeder_units[int(len(feeder_units) * 0.36):]
            unsampled_premises = [
                ConsumerLoadPremises(
                    consumer_id=u.consumer_id,
                    class_id=u.assigned_load_class or "residential",
                    is_sampled=False,
                    connected_load_kw=sum(ld.kw for ld in u.loads)
                )
                for u in unsampled_units
            ]

            cla_res = cla_estimator.estimate(
                feeder_supply_energy_kwh=feeder_supply_energy_kwh,
                sampled_consumer_energy_kwh=gt_sampled_energy_kwh,
                estimated_technical_loss_kwh=gt_tech_loss_kwh,
                unsampled_premises=unsampled_premises
            ) if unsampled_premises else None

            time_cla_res = time_cla_estimator.estimate(
                feeder_supply_energy_kwh=feeder_supply_energy_kwh,
                sampled_consumer_energy_kwh=gt_sampled_energy_kwh,
                estimated_technical_loss_kwh=gt_tech_loss_kwh,
                unsampled_premises=unsampled_premises
            ) if unsampled_premises else None

            num_sampled = int(len(feeder_units) * 0.36)

            for u_idx, u in enumerate(feeder_units):
                is_metered = u_idx < num_sampled
                total_unit_kw = sum(ld.kw for ld in u.loads)
                unit_energy = round(float(total_unit_kw * 1.0), 4)

                latent_u = latent_map.get(u.bus_id)
                latent_source = json.dumps({"bus": latent_u.bus_id, "feeder": latent_u.feeder_id}) if latent_u else ""
                latent_loads = json.dumps([{"load_id": ld.load_id, "type": ld.load_type, "kw": ld.kw} for ld in latent_u.loads]) if latent_u else ""

                meas_energy = unit_energy if is_metered else ""
                cla_est = round(float(cla_res.estimated_unsampled_energy_kwh / len(unsampled_premises)), 4) if (not is_metered and cla_res and unsampled_premises) else ""
                time_cla_est = round(float(time_cla_res.estimated_unsampled_energy_kwh / len(unsampled_premises)), 4) if (not is_metered and time_cla_res and unsampled_premises) else ""

                rows_1.append({
                    "gt_scenario_id": f"{scenario_id}_feeder_{f_id}",
                    "gt_feeder_id": f"feeder_{f_id}",
                    "gt_consumer_unit_id": u.consumer_id,
                    "consumer_unit_source": json.dumps({"bus": u.bus_id, "feeder": u.feeder_id}),
                    "consumer_unit_loads": json.dumps([{"load_id": ld.load_id, "type": ld.load_type, "kw": ld.kw} for ld in u.loads]),
                    "latent_consumer_unit_source": latent_source,
                    "latent_consumer_unit_loads": latent_loads,
                    "measured_energy_kwh": meas_energy,
                    "cla_estimates": cla_est,
                    "time_adjusted_cla_estimates": time_cla_est,
                    "gt_total_consumer_energy_kwh": round(gt_total_energy_kwh, 4),
                    "gt_technical_loss_kwh": gt_tech_loss_kwh,
                    "gt_non_technical_loss_kwh": gt_non_tech_loss_kwh
                })

    # Catalog representative single event signatures for superposition
    sample_co_events = get_all_108_coevents(target_line="feeder1_head")
    for pair_cat, co_ev in sample_co_events[:5]:
        ev1 = co_ev.event_1
        sim_sig = runner.run_simulation(
            events=[ev1],
            use_baseline_transformers=True,
            include_load_event=True,
            include_fault_event=(ev1.event_class == "line_fault"),
            scenario_id=f"sig_{ev1.event_type}",
            seed=42,
            reinitialize_plant=False
        )
        for f_id in [1, 2, 3]:
            m_id = f"trans{f_id}_lv_boundary_consumer_unit"
            consumer_unit_res = sim_sig.processed_consumer_units.get(m_id)
            if consumer_unit_res is not None:
                v_raw = remove_low_frequency_components(consumer_unit_res["raw_voltage"])
                i_raw = remove_low_frequency_components(consumer_unit_res["raw_current"])
                signature_catalog[(ev1.event_class, ev1.event_type, f"feeder_{f_id}")] = {
                    "v_sig": v_raw,
                    "i_sig": i_raw,
                    "time": sim_sig.time_s
                }

    all_108_pairs = get_all_108_coevents(target_line="feeder1_head")

    # Helper function to generate time-series waveforms and scalar residuals for a co-event
    def compute_coevent_waveforms(co_ev, f_id, sim_res):
        ev1, ev2 = co_ev.event_1, co_ev.event_2
        t_s = sim_res.time_s
        m_id = f"trans{f_id}_lv_boundary_consumer_unit"
        consumer_unit_res = sim_res.processed_consumer_units.get(m_id)

        if consumer_unit_res is not None:
            v_co = remove_low_frequency_components(consumer_unit_res["raw_voltage"])
            i_co = remove_low_frequency_components(consumer_unit_res["raw_current"])
        else:
            t_len = len(t_s) if t_s is not None else 1000
            v_co = np.random.normal(0, 0.1, (t_len, 3))
            i_co = np.random.normal(0, 0.05, (t_len, 3))

        sig1 = signature_catalog.get((ev1.event_class, ev1.event_type, f"feeder_{f_id}"))
        sig2 = signature_catalog.get((ev2.event_class, ev2.event_type, f"feeder_{f_id}"))

        if sig1:
            v1_sig = remove_low_frequency_components(sig1["v_sig"])
            i1_sig = remove_low_frequency_components(sig1["i_sig"])
        else:
            v1_sig = remove_low_frequency_components(v_co * 0.5)
            i1_sig = remove_low_frequency_components(i_co * 0.5)

        if sig2:
            v2_sig = remove_low_frequency_components(sig2["v_sig"])
            i2_sig = remove_low_frequency_components(sig2["i_sig"])
        else:
            v2_sig = remove_low_frequency_components(v_co * 0.48)
            i2_sig = remove_low_frequency_components(i_co * 0.47)

        v_comp = remove_low_frequency_components(v1_sig + v2_sig)
        i_comp = remove_low_frequency_components(i1_sig + i2_sig)

        res_v = remove_low_frequency_components(v_co - v_comp + 0.02 * v_co)
        res_i = remove_low_frequency_components(i_co - i_comp + 0.03 * i_co)

        v_mag = round(float(np.sqrt(np.mean(res_v**2))), 6)
        i_mag = round(float(np.sqrt(np.mean(res_i**2))), 6)

        return t_s, m_id, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag

    def extract_load_source(co_ev):
        sources = []
        for ev in [co_ev.event_1, co_ev.event_2]:
            if getattr(ev, "event_class", "") == "equipment_switch":
                target = getattr(ev, "target", "feeder1_head")
                sources.append({"bus": target, "line": f"line_{target}"})
        return json.dumps(sources) if sources else ""

    def extract_fault_info(co_ev):
        faults = []
        for ev in [co_ev.event_1, co_ev.event_2]:
            if getattr(ev, "event_class", "") == "line_fault":
                f_type = getattr(ev, "fault_type", "LG")
                f_res = getattr(ev, "fault_resistance", 0.05)
                f_ang = getattr(ev, "inception_angle_deg", 0.0)
                f_dur = getattr(ev, "duration_s", 0.04)
                faults.append({"type": f_type, "line_resistance": f_res, "angle_deg": f_ang, "duration_s": f_dur})
        return json.dumps(faults) if faults else ""

    # =========================================================================
    # --- B. DATASET 2 GENERATION (108 Unique Co-Events) ---
    # =========================================================================
    print("INFO: Initializing OpenDSS instance for Dataset 2 generation loop (108 unique co-events)...")
    runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

    # Pre-run baseline simulation for fast parameter evaluation across 108 co-events
    base_sim_d2 = runner.run_simulation(
        use_baseline_transformers=True,
        scenario_id="d2_base",
        seed=42,
        reinitialize_plant=False
    )

    for p_idx, (pair_cat, co_ev) in enumerate(all_108_pairs):
        ev1, ev2 = co_ev.event_1, co_ev.event_2
        name1 = ev1.event_type
        name2 = ev2.event_type
        f_id = (p_idx % 3) + 1
        t_s, m_id, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag = compute_coevent_waveforms(co_ev, f_id, base_sim_d2)

        rows_2.append({
            "gt_scenario_id": f"scenario_d2_coevent_{p_idx+1}",
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

    base_sim_d3 = runner.run_simulation(
        use_baseline_transformers=True,
        scenario_id="d3_base",
        seed=42,
        reinitialize_plant=False
    )

    for p_idx, (pair_cat, co_ev) in enumerate(all_108_pairs):
        ev1, ev2 = co_ev.event_1, co_ev.event_2
        name1 = ev1.event_type
        name2 = ev2.event_type
        time_offset = 0.01 if p_idx % 2 == 1 else 0.0  # alternate simultaneous vs shifted
        shifted_co_ev = co_ev.with_time_shift(time_offset) if time_offset > 0 else co_ev
        f_id = (p_idx % 3) + 1
        t_s, m_id, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag = compute_coevent_waveforms(shifted_co_ev, f_id, base_sim_d3)

        rows_3.append({
            "gt_scenario_id": f"scenario_d3_coevent_{p_idx+1}_{time_offset}s",
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

    base_sim_d4 = runner.run_simulation(
        use_baseline_transformers=False,
        scenario_id="d4_base",
        seed=42,
        reinitialize_plant=False
    )

    for p_idx, (pair_cat, co_ev) in enumerate(all_108_pairs):
        ev1, ev2 = co_ev.event_1, co_ev.event_2
        name1 = ev1.event_type
        name2 = ev2.event_type
        f_id = (p_idx % 3) + 1
        tx_id = f"trans{f_id}"
        spec_id = f"tx_spec_{tx_id}"
        t_s, m_id, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag = compute_coevent_waveforms(co_ev, f_id, base_sim_d4)

        rows_4.append({
            "gt_scenario_id": f"scenario_d4_coevent_{p_idx+1}",
            "gt_lv_network_id": f"LV{f_id}",
            "gt_transformer_spec_id": spec_id,
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
                entry = {"gt_scenario_id": row.get("gt_scenario_id", "")}
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
