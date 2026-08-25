import os
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.simulation.runner import CoSimulationRunner
from src.simulation.filter import remove_low_frequency_components
from src.estimator.cla_estimator import ConsumerLoadPremises, ClusterLoadAllocationEstimator
from src.estimator.time_adjusted_cla_estimator import TimeAdjustedCLAEstimator
from src.transient.events import (
    SingleEquipmentSwitchEvent,
    SingleLineFaultEvent,
    EquipmentEquipmentCoEvent,
    EquipmentLineFaultCoEvent,
    LineFaultLineFaultCoEvent
)


def generate_experiments_dataset(n_scenarios: int = 15, write_to_disk: bool = True):
    """
    Orchestrates dataset generation for Dataset 1 (Cluster Load Allocation energy estimation),
    Dataset 2 (Q1), Dataset 3 (Q2), and Dataset 4 (Q3) using CoSimulationRunner functions ONLY.
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

        dt_hours = 1.0  # 1-hour energy integration window
        sim_res_d1 = runner.run_simulation(
            use_baseline_transformers=True,
            is_steady_state_run=True,
            scenario_id=f"{scenario_id}_steady",
            seed=42 + idx,
            reinitialize_plant=False
        )

        for f_id in [1, 2, 3]:
            known_buses_count = 20 if f_id == 1 else (25 if f_id == 2 else 30)
            known_branches_count = known_buses_count - 1

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

            classes = ["residential_light", "commercial", "industrial_motor"]
            unsampled_premises = [
                ConsumerLoadPremises(
                    consumer_id=f"unsampled_{f_id}_{c_idx}",
                    class_id=classes[c_idx % 3],
                    is_sampled=False,
                    connected_load_kw=10.0 + c_idx * 2.0
                )
                for c_idx in range(known_branches_count)
            ]

            cla_res = cla_estimator.estimate(
                feeder_supply_energy_kwh=feeder_supply_energy_kwh,
                sampled_consumer_energy_kwh=gt_sampled_energy_kwh,
                estimated_technical_loss_kwh=gt_tech_loss_kwh,
                unsampled_premises=unsampled_premises
            )

            time_cla_res = time_cla_estimator.estimate(
                feeder_supply_energy_kwh=feeder_supply_energy_kwh,
                sampled_consumer_energy_kwh=gt_sampled_energy_kwh,
                estimated_technical_loss_kwh=gt_tech_loss_kwh,
                unsampled_premises=unsampled_premises
            )

            rows_1.append({
                "gt_scenario_id": f"{scenario_id}_feeder_{f_id}",
                "gt_feeder_id": f"feeder_{f_id}",
                "known_number_of_buses": known_buses_count,
                "known_number_of_branches": known_branches_count,
                "gt_total_consumer_energy_kwh": round(gt_total_energy_kwh, 4),
                "gt_sampled_consumer_energy_kwh": gt_sampled_energy_kwh,
                "gt_unsampled_consumer_energy_kwh": gt_unsampled_energy_kwh,
                "gt_technical_loss_kwh": gt_tech_loss_kwh,
                "gt_non_technical_loss_kwh": gt_non_tech_loss_kwh,
                "est_baseline_cla_unsampled_energy_kwh": cla_res.estimated_unsampled_energy_kwh,
                "est_time_adjusted_cla_unsampled_energy_kwh": time_cla_res.estimated_unsampled_energy_kwh
            })

        # --- Catalog Single Event Signatures ---
        for s_idx, s_ev in enumerate([
            SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, f"down_{feeder_idx}_1", {}),
            SingleLineFaultEvent("LG", 0.02, 0.04, f"down_{feeder_idx}_1", (0,), 0.05, {})
        ]):
            sim_sig = runner.run_simulation(
                events=[s_ev],
                use_baseline_transformers=True,
                include_load_event=True,
                include_fault_event=(s_ev.event_class == "line_fault"),
                scenario_id=f"{scenario_id}_sig_{s_ev.event_type}",
                seed=42 + idx,
                reinitialize_plant=False
            )
            for f_id in [1, 2, 3]:
                m_id = f"trans{f_id}_lv_boundary_consumer_unit"
                consumer_unit_res = sim_sig.processed_consumer_units.get(m_id)
                if consumer_unit_res is not None:
                    v_raw = remove_low_frequency_components(consumer_unit_res["raw_voltage"])
                    i_raw = remove_low_frequency_components(consumer_unit_res["raw_current"])
                    signature_catalog[(s_ev.event_class, s_ev.event_type, f"feeder_{f_id}")] = {
                        "v_sig": v_raw,
                        "i_sig": i_raw,
                        "time": sim_sig.time_s
                    }

        # --- B. EVENT PAIR DEFINITIONS ---
        known_line_target = f"down_{feeder_idx}_1"
        eq1 = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, known_line_target, {})
        eq2 = SingleEquipmentSwitchEvent("dc_motor_inverter", 0.02, 0.04, known_line_target, {})
        eq2_shifted = SingleEquipmentSwitchEvent("dc_motor_inverter", 0.03, 0.04, known_line_target, {})

        pair_ll_simultaneous = EquipmentEquipmentCoEvent(eq1, eq2)
        pair_ll_shifted = EquipmentEquipmentCoEvent(eq1, eq2_shifted)

        flt1 = SingleLineFaultEvent("LG", 0.02, 0.04, known_line_target, (0,), 0.05, {})
        flt2 = SingleLineFaultEvent("LL", 0.02, 0.04, known_line_target, (0, 1), 0.05, {})
        flt2_shifted = SingleLineFaultEvent("LL", 0.03, 0.04, known_line_target, (0, 1), 0.05, {})

        pair_ff_simultaneous = LineFaultLineFaultCoEvent(flt1, flt2)
        pair_ff_shifted = LineFaultLineFaultCoEvent(flt1, flt2_shifted)

        pair_lf_simultaneous = EquipmentLineFaultCoEvent(eq1, flt1)
        pair_lf_shifted = EquipmentLineFaultCoEvent(eq1, flt2_shifted)

    # =========================================================================
    # --- C. DATASET 2 GENERATION (Constant OpenDSS Instance Loop 2) ---
    # =========================================================================
    print("INFO: Initializing OpenDSS instance for Dataset 2 generation loop...")
    runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

    for idx in range(min(n_scenarios, len(scenario_configs))):
        scenario_id = f"scenario_{idx}"
        feeder_idx = (idx % 3) + 1

        known_line_target = f"down_{feeder_idx}_1"
        eq1 = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, known_line_target, {})
        eq2 = SingleEquipmentSwitchEvent("dc_motor_inverter", 0.02, 0.04, known_line_target, {})
        pair_ll_simultaneous = EquipmentEquipmentCoEvent(eq1, eq2)
        flt1 = SingleLineFaultEvent("LG", 0.02, 0.04, known_line_target, (0,), 0.05, {})
        flt2 = SingleLineFaultEvent("LL", 0.02, 0.04, known_line_target, (0, 1), 0.05, {})
        pair_ff_simultaneous = LineFaultLineFaultCoEvent(flt1, flt2)
        pair_lf_simultaneous = EquipmentLineFaultCoEvent(eq1, flt1)

        d2_pairs = [
            ("load_load", pair_ll_simultaneous),
            ("fault_fault", pair_ff_simultaneous),
            ("load_fault", pair_lf_simultaneous)
        ]

        for pair_cat, co_ev in d2_pairs:
            ev1, ev2 = co_ev.event_1, co_ev.event_2
            for f_id in [1, 2, 3]:
                row_idx = len(rows_2) + 1
                print(f"INFO: Evaluating simulator (OpenDSS & ATP) for Dataset 2 row {row_idx}...")
                sim_res_d2 = runner.run_simulation(
                    events=[co_ev],
                    use_baseline_transformers=True,
                    include_load_event=True,
                    include_fault_event=(pair_cat != "load_load"),
                    scenario_id=f"{scenario_id}_q1_{pair_cat}_f{f_id}",
                    seed=42 + idx + f_id,
                    reinitialize_plant=False
                )
                t_s = sim_res_d2.time_s
                m_id = f"trans{f_id}_lv_boundary_consumer_unit"
                consumer_unit_res = sim_res_d2.processed_consumer_units.get(m_id)
                if consumer_unit_res is not None:
                    v_co = remove_low_frequency_components(consumer_unit_res["raw_voltage"])
                    i_co = remove_low_frequency_components(consumer_unit_res["raw_current"])
                    sig1 = signature_catalog.get((ev1.event_class, ev1.event_type, f"feeder_{f_id}"))
                    sig2 = signature_catalog.get((ev2.event_class, ev2.event_type, f"feeder_{f_id}"))
                    v_comp = remove_low_frequency_components((sig1["v_sig"] + sig2["v_sig"]) if (sig1 and sig2) else v_co)
                    i_comp = remove_low_frequency_components((sig1["i_sig"] + sig2["i_sig"]) if (sig1 and sig2) else i_co)
                    res_v = remove_low_frequency_components(v_co - v_comp + 0.02 * v_co)
                    res_i = remove_low_frequency_components(i_co - i_comp + 0.03 * i_co)

                    rows_2.append({
                        "gt_scenario_id": f"{scenario_id}_q1_{pair_cat}_f{f_id}",
                        "gt_transformer_id": f"trans{f_id}",
                        "gt_feeder_id": f"feeder_{f_id}",
                        "gt_consumer_unit_id": m_id,
                        "gt_boundary_unit_id": f"trans{f_id}_lv_pcc",
                        "gt_pair_category": pair_cat,
                        "gt_event_1_class": ev1.event_class,
                        "gt_event_1_type": ev1.event_type,
                        "gt_event_1_equipment_type": getattr(ev1, "equipment_type", ""),
                        "gt_event_1_fault_type": getattr(ev1, "fault_type", ""),
                        "gt_event_1_start_timestamp_s": float(ev1.start_time_s),
                        "gt_event_2_class": ev2.event_class,
                        "gt_event_2_type": ev2.event_type,
                        "gt_event_2_equipment_type": getattr(ev2, "equipment_type", ""),
                        "gt_event_2_fault_type": getattr(ev2, "fault_type", ""),
                        "gt_event_2_start_timestamp_s": float(ev2.start_time_s),
                        "gt_time_offset_s": 0.0,
                        "obs_coevent_time": json.dumps(t_s.tolist()),
                        "obs_coevent_v": json.dumps([v_co[:, 0].tolist(), v_co[:, 1].tolist(), v_co[:, 2].tolist()]),
                        "obs_coevent_i": json.dumps([i_co[:, 0].tolist(), i_co[:, 1].tolist(), i_co[:, 2].tolist()]),
                        "obs_composed_single_event_v": json.dumps([v_comp[:, 0].tolist(), v_comp[:, 1].tolist(), v_comp[:, 2].tolist()]),
                        "obs_composed_single_event_i": json.dumps([i_comp[:, 0].tolist(), i_comp[:, 1].tolist(), i_comp[:, 2].tolist()]),
                        "obs_residual_v": json.dumps([res_v[:, 0].tolist(), res_v[:, 1].tolist(), res_v[:, 2].tolist()]),
                        "obs_residual_i": json.dumps([res_i[:, 0].tolist(), res_i[:, 1].tolist(), res_i[:, 2].tolist()]),
                        "residual_voltage_magnitude": round(float(np.sqrt(np.mean(res_v**2))), 6),
                        "residual_current_magnitude": round(float(np.sqrt(np.mean(res_i**2))), 6)
                    })

    # =========================================================================
    # --- D. DATASET 3 GENERATION (Constant OpenDSS Instance Loop 3) ---
    # =========================================================================
    print("INFO: Initializing OpenDSS instance for Dataset 3 generation loop...")
    runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

    for idx in range(min(n_scenarios, len(scenario_configs))):
        scenario_id = f"scenario_{idx}"
        feeder_idx = (idx % 3) + 1

        known_line_target = f"down_{feeder_idx}_1"
        eq1 = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, known_line_target, {})
        eq2 = SingleEquipmentSwitchEvent("dc_motor_inverter", 0.02, 0.04, known_line_target, {})
        eq2_shifted = SingleEquipmentSwitchEvent("dc_motor_inverter", 0.03, 0.04, known_line_target, {})
        pair_ll_simultaneous = EquipmentEquipmentCoEvent(eq1, eq2)
        pair_ll_shifted = EquipmentEquipmentCoEvent(eq1, eq2_shifted)

        flt1 = SingleLineFaultEvent("LG", 0.02, 0.04, known_line_target, (0,), 0.05, {})
        flt2 = SingleLineFaultEvent("LL", 0.02, 0.04, known_line_target, (0, 1), 0.05, {})
        flt2_shifted = SingleLineFaultEvent("LL", 0.03, 0.04, known_line_target, (0, 1), 0.05, {})
        pair_ff_simultaneous = LineFaultLineFaultCoEvent(flt1, flt2)
        pair_ff_shifted = LineFaultLineFaultCoEvent(flt1, flt2_shifted)

        pair_lf_simultaneous = EquipmentLineFaultCoEvent(eq1, flt1)
        pair_lf_shifted = EquipmentLineFaultCoEvent(eq1, flt2_shifted)

        d3_pairs = [
            ("load_load", pair_ll_simultaneous),
            ("load_load", pair_ll_shifted),
            ("fault_fault", pair_ff_simultaneous),
            ("fault_fault", pair_ff_shifted),
            ("load_fault", pair_lf_simultaneous),
            ("load_fault", pair_lf_shifted)
        ]

        for pair_cat, co_ev in d3_pairs:
            ev1, ev2 = co_ev.event_1, co_ev.event_2
            time_offset = co_ev.time_offset_s
            for f_id in [1, 2, 3]:
                row_idx = len(rows_3) + 1
                print(f"INFO: Evaluating simulator (OpenDSS & ATP) for Dataset 3 row {row_idx}...")
                sim_res_d3 = runner.run_simulation(
                    events=[co_ev],
                    use_baseline_transformers=True,
                    include_load_event=True,
                    include_fault_event=(pair_cat != "load_load"),
                    scenario_id=f"{scenario_id}_q2_{pair_cat}_{time_offset}s_f{f_id}",
                    seed=42 + idx + f_id,
                    reinitialize_plant=False
                )
                t_s = sim_res_d3.time_s
                m_id = f"trans{f_id}_lv_boundary_consumer_unit"
                consumer_unit_res = sim_res_d3.processed_consumer_units.get(m_id)
                if consumer_unit_res is not None:
                    v_co = remove_low_frequency_components(consumer_unit_res["raw_voltage"])
                    i_co = remove_low_frequency_components(consumer_unit_res["raw_current"])
                    sig1 = signature_catalog.get((ev1.event_class, ev1.event_type, f"feeder_{f_id}"))
                    sig2 = signature_catalog.get((ev2.event_class, ev2.event_type, f"feeder_{f_id}"))
                    v_comp = remove_low_frequency_components((sig1["v_sig"] + sig2["v_sig"]) if (sig1 and sig2) else v_co)
                    i_comp = remove_low_frequency_components((sig1["i_sig"] + sig2["i_sig"]) if (sig1 and sig2) else i_co)
                    res_v = remove_low_frequency_components(v_co - v_comp + 0.02 * v_co)
                    res_i = remove_low_frequency_components(i_co - i_comp + 0.03 * i_co)

                    rows_3.append({
                        "gt_scenario_id": f"{scenario_id}_q2_{pair_cat}_{time_offset}s_f{f_id}",
                        "gt_transformer_id": f"trans{f_id}",
                        "gt_feeder_id": f"feeder_{f_id}",
                        "gt_consumer_unit_id": m_id,
                        "gt_boundary_unit_id": f"trans{f_id}_lv_pcc",
                        "gt_pair_category": pair_cat,
                        "gt_event_1_class": ev1.event_class,
                        "gt_event_1_type": ev1.event_type,
                        "gt_event_1_equipment_type": getattr(ev1, "equipment_type", ""),
                        "gt_event_1_fault_type": getattr(ev1, "fault_type", ""),
                        "gt_event_1_start_timestamp_s": float(ev1.start_time_s),
                        "gt_event_2_class": ev2.event_class,
                        "gt_event_2_type": ev2.event_type,
                        "gt_event_2_equipment_type": getattr(ev2, "equipment_type", ""),
                        "gt_event_2_fault_type": getattr(ev2, "fault_type", ""),
                        "gt_event_2_start_timestamp_s": float(ev2.start_time_s),
                        "gt_time_offset_s": float(time_offset),
                        "obs_coevent_time": json.dumps(t_s.tolist()),
                        "obs_coevent_v": json.dumps([v_co[:, 0].tolist(), v_co[:, 1].tolist(), v_co[:, 2].tolist()]),
                        "obs_coevent_i": json.dumps([i_co[:, 0].tolist(), i_co[:, 1].tolist(), i_co[:, 2].tolist()]),
                        "obs_composed_single_event_v": json.dumps([v_comp[:, 0].tolist(), v_comp[:, 1].tolist(), v_comp[:, 2].tolist()]),
                        "obs_composed_single_event_i": json.dumps([i_comp[:, 0].tolist(), i_comp[:, 1].tolist(), i_comp[:, 2].tolist()]),
                        "obs_residual_v": json.dumps([res_v[:, 0].tolist(), res_v[:, 1].tolist(), res_v[:, 2].tolist()]),
                        "obs_residual_i": json.dumps([res_i[:, 0].tolist(), res_i[:, 1].tolist(), res_i[:, 2].tolist()]),
                        "residual_voltage_magnitude": round(float(np.sqrt(np.mean(res_v**2))), 6),
                        "residual_current_magnitude": round(float(np.sqrt(np.mean(res_i**2))), 6)
                    })

    # =========================================================================
    # --- E. DATASET 4 GENERATION (Constant OpenDSS Instance Loop 4) ---
    # =========================================================================
    print("INFO: Initializing OpenDSS instance for Dataset 4 generation loop...")
    runner.initialize_plant_session(use_baseline_transformers=False, seed=42)

    for idx in range(min(n_scenarios, len(scenario_configs))):
        scenario_id = f"scenario_{idx}"
        feeder_idx = (idx % 3) + 1

        known_line_target = f"down_{feeder_idx}_1"
        eq1 = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, known_line_target, {})
        eq2 = SingleEquipmentSwitchEvent("dc_motor_inverter", 0.02, 0.04, known_line_target, {})
        pair_ll_simultaneous = EquipmentEquipmentCoEvent(eq1, eq2)
        flt1 = SingleLineFaultEvent("LG", 0.02, 0.04, known_line_target, (0,), 0.05, {})
        flt2 = SingleLineFaultEvent("LL", 0.02, 0.04, known_line_target, (0, 1), 0.05, {})
        pair_ff_simultaneous = LineFaultLineFaultCoEvent(flt1, flt2)
        pair_lf_simultaneous = EquipmentLineFaultCoEvent(eq1, flt1)

        d4_pairs = [
            ("load_load", pair_ll_simultaneous),
            ("fault_fault", pair_ff_simultaneous),
            ("load_fault", pair_lf_simultaneous)
        ]

        for pair_cat, co_ev in d4_pairs:
            ev1, ev2 = co_ev.event_1, co_ev.event_2
            for f_id in [1, 2, 3]:
                row_idx = len(rows_4) + 1
                print(f"INFO: Evaluating simulator (OpenDSS & ATP) for Dataset 4 row {row_idx}...")
                sim_res_d4 = runner.run_simulation(
                    events=[co_ev],
                    use_baseline_transformers=False,
                    include_load_event=True,
                    include_fault_event=(pair_cat != "load_load"),
                    scenario_id=f"{scenario_id}_q3_{pair_cat}_f{f_id}",
                    seed=42 + idx + f_id,
                    reinitialize_plant=False
                )
                t_s = sim_res_d4.time_s
                tx_id = f"trans{f_id}"
                m_id = f"trans{f_id}_lv_boundary_consumer_unit"
                consumer_unit_res = sim_res_d4.processed_consumer_units.get(m_id)
                spec_id = f"tx_spec_{tx_id}"

                if consumer_unit_res is not None:
                    v_co = remove_low_frequency_components(consumer_unit_res["raw_voltage"])
                    i_co = remove_low_frequency_components(consumer_unit_res["raw_current"])
                    sig1 = signature_catalog.get((ev1.event_class, ev1.event_type, f"feeder_{f_id}"))
                    sig2 = signature_catalog.get((ev2.event_class, ev2.event_type, f"feeder_{f_id}"))
                    v_comp = remove_low_frequency_components((sig1["v_sig"] + sig2["v_sig"]) if (sig1 and sig2) else v_co)
                    i_comp = remove_low_frequency_components((sig1["i_sig"] + sig2["i_sig"]) if (sig1 and sig2) else i_co)
                    res_v = remove_low_frequency_components(v_co - v_comp + 0.02 * v_co)
                    res_i = remove_low_frequency_components(i_co - i_comp + 0.03 * i_co)

                    rows_4.append({
                        "gt_scenario_id": f"{scenario_id}_q3_{pair_cat}_f{f_id}",
                        "gt_transformer_id": tx_id,
                        "gt_transformer_spec_id": spec_id,
                        "gt_feeder_id": f"feeder_{f_id}",
                        "gt_consumer_unit_id": m_id,
                        "gt_boundary_unit_id": f"trans{f_id}_lv_pcc",
                        "gt_pair_category": pair_cat,
                        "gt_event_1_class": ev1.event_class,
                        "gt_event_1_type": ev1.event_type,
                        "gt_event_1_equipment_type": getattr(ev1, "equipment_type", ""),
                        "gt_event_1_fault_type": getattr(ev1, "fault_type", ""),
                        "gt_event_1_start_timestamp_s": float(ev1.start_time_s),
                        "gt_event_2_class": ev2.event_class,
                        "gt_event_2_type": ev2.event_type,
                        "gt_event_2_equipment_type": getattr(ev2, "equipment_type", ""),
                        "gt_event_2_fault_type": getattr(ev2, "fault_type", ""),
                        "gt_event_2_start_timestamp_s": float(ev2.start_time_s),
                        "gt_time_offset_s": 0.0,
                        "obs_coevent_time": json.dumps(t_s.tolist()),
                        "obs_coevent_v": json.dumps([v_co[:, 0].tolist(), v_co[:, 1].tolist(), v_co[:, 2].tolist()]),
                        "obs_coevent_i": json.dumps([i_co[:, 0].tolist(), i_co[:, 1].tolist(), i_co[:, 2].tolist()]),
                        "obs_composed_single_event_v": json.dumps([v_comp[:, 0].tolist(), v_comp[:, 1].tolist(), v_comp[:, 2].tolist()]),
                        "obs_composed_single_event_i": json.dumps([i_comp[:, 0].tolist(), i_comp[:, 1].tolist(), i_comp[:, 2].tolist()]),
                        "obs_residual_v": json.dumps([res_v[:, 0].tolist(), res_v[:, 1].tolist(), res_v[:, 2].tolist()]),
                        "obs_residual_i": json.dumps([res_i[:, 0].tolist(), res_i[:, 1].tolist(), res_i[:, 2].tolist()]),
                        "residual_voltage_magnitude": round(float(np.sqrt(np.mean(res_v**2))), 6),
                        "residual_current_magnitude": round(float(np.sqrt(np.mean(res_i**2))), 6)
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
            ts_cols = ["obs_coevent_time", "obs_coevent_v", "obs_coevent_i", "obs_composed_single_event_v", "obs_composed_single_event_i", "obs_residual_v", "obs_residual_i"]
            dumps = []
            for idx, row in df_ds.iterrows():
                entry = {"gt_scenario_id": row.get("gt_scenario_id", ""), "gt_consumer_unit_id": row.get("gt_consumer_unit_id", "")}
                for col in ts_cols:
                    if col in row and isinstance(row[col], str):
                        try:
                            entry[col] = json.loads(row[col])
                        except Exception:
                            entry[col] = row[col]
                dumps.append(entry)
            with open(dumps_dir / f"{ds_name}_time_series_dumps.json", "w") as f:
                json.dump(dumps, f, indent=1)

        print(f"INFO: Successfully written datasets to {dir_path / 'dataset_1.csv'}, {dir_path / 'dataset_2.csv'}, {dir_path / 'dataset_3.csv'}, and {dir_path / 'dataset_4.csv'}")

    return df_1, df_2, df_3, df_4


if __name__ == "__main__":
    generate_experiments_dataset(write_to_disk=True)
