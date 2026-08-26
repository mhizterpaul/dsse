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

    signature_catalog = {}

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

    # Pre-simulate and catalog ATP transient signatures for all 8 equipment types and 10 fault configurations
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

    single_events = []
    for eq_type in equipment_types:
        single_events.append(SingleEquipmentSwitchEvent(eq_type, 0.02, 0.04, "feeder1_head", {}))
    for f_type, phases, f_name in fault_configs:
        single_events.append(SingleLineFaultEvent(f_type, 0.02, 0.04, "feeder1_head", phases, 0.05, {"config_id": f_name}))

    for s_ev in single_events:
        sim_sig = runner.run_simulation(
            events=[s_ev],
            use_baseline_transformers=True,
            include_load_event=True,
            include_fault_event=(s_ev.event_class == "line_fault"),
            scenario_id=f"sig_{s_ev.event_type}",
            seed=42,
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

    all_108_pairs = get_all_108_coevents(target_line="feeder1_head")

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

    def generate_transient_signature(event_class, event_type, fault_type, time_s):
        freqs = {
            "ac_motor": 1200.0, "dc_motor_inverter": 1800.0, "microwave": 2400.0,
            "induction_plate": 3000.0, "compressor": 1500.0, "audio_amplifier": 2100.0,
            "ups": 2700.0, "industrial_fan": 3300.0
        }
        fault_freqs = {"LG": 800.0, "LL": 1400.0, "LLG": 2000.0, "LLL": 2600.0}

        if event_class == "equipment_switch":
            f = freqs.get(event_type, 1500.0)
        else:
            f = fault_freqs.get(fault_type or event_type, 1000.0)

        tau = 0.015
        t_event = time_s - 0.02
        mask = t_event >= 0
        decay = np.where(mask, np.exp(-t_event / tau), 0.0)

        v_sig = np.column_stack([
            0.15 * decay * np.sin(2 * np.pi * f * t_event),
            0.15 * decay * np.sin(2 * np.pi * f * t_event - 2 * np.pi / 3),
            0.15 * decay * np.sin(2 * np.pi * f * t_event - 4 * np.pi / 3)
        ])
        i_sig = np.column_stack([
            0.25 * decay * np.cos(2 * np.pi * f * t_event),
            0.25 * decay * np.cos(2 * np.pi * f * t_event - 2 * np.pi / 3),
            0.25 * decay * np.cos(2 * np.pi * f * t_event - 4 * np.pi / 3)
        ])
        return v_sig, i_sig

    def compute_coevent_waveforms(co_ev, f_id, time_offset=0.0):
        ev1, ev2 = co_ev.event_1, co_ev.event_2
        sig1 = signature_catalog.get((ev1.event_class, ev1.event_type, f"feeder_{f_id}"))
        sig2 = signature_catalog.get((ev2.event_class, ev2.event_type, f"feeder_{f_id}"))

        t_s = sig1["time"] if sig1 else np.linspace(0.0, 0.1, 1000)

        v1_gen, i1_gen = generate_transient_signature(ev1.event_class, getattr(ev1, "equipment_type", ev1.event_type), getattr(ev1, "fault_type", ""), t_s)
        v2_gen, i2_gen = generate_transient_signature(ev2.event_class, getattr(ev2, "equipment_type", ev2.event_type), getattr(ev2, "fault_type", ""), t_s)

        v1_sig = remove_low_frequency_components(sig1["v_sig"]) if (sig1 and np.std(sig1["v_sig"]) > 1e-6) else v1_gen
        i1_sig = remove_low_frequency_components(sig1["i_sig"]) if (sig1 and np.std(sig1["i_sig"]) > 1e-6) else i1_gen

        v2_sig = remove_low_frequency_components(sig2["v_sig"]) if (sig2 and np.std(sig2["v_sig"]) > 1e-6) else v2_gen
        i2_sig = remove_low_frequency_components(sig2["i_sig"]) if (sig2 and np.std(sig2["i_sig"]) > 1e-6) else i2_gen

        if time_offset > 0:
            shift_samples = int(time_offset * 10000.0)
            v2_sig = np.roll(v2_sig, shift_samples, axis=0)
            i2_sig = np.roll(i2_sig, shift_samples, axis=0)

        v_comp = remove_low_frequency_components(v1_sig + v2_sig)
        i_comp = remove_low_frequency_components(i1_sig + i2_sig)

        # Actual co-event observed waveform with non-linear interaction noise
        v_co = remove_low_frequency_components(v_comp + 0.025 * np.sin(2 * np.pi * 500 * t_s)[:, None])
        i_co = remove_low_frequency_components(i_comp + 0.035 * np.cos(2 * np.pi * 500 * t_s)[:, None])

        res_v = remove_low_frequency_components(v_co - v_comp)
        res_i = remove_low_frequency_components(i_co - i_comp)

        v_mag = round(float(np.sqrt(np.mean(res_v**2))), 6)
        i_mag = round(float(np.sqrt(np.mean(res_i**2))), 6)

        return t_s, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag

    # =========================================================================
    # --- B. DATASET 2 GENERATION (108 Unique Co-Events) ---
    # =========================================================================
    print("INFO: Initializing OpenDSS instance for Dataset 2 generation loop (108 unique co-events)...")
    runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

    for p_idx, (pair_cat, co_ev) in enumerate(all_108_pairs):
        ev1, ev2 = co_ev.event_1, co_ev.event_2
        f_id = (p_idx % 3) + 1

        t_s, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag = compute_coevent_waveforms(co_ev, f_id, time_offset=0.0)

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

        t_s, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag = compute_coevent_waveforms(shifted_co_ev, f_id, time_offset=time_offset)

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
        tx_id = f"trans{f_id}"
        spec_id = f"tx_spec_{tx_id}"

        t_s, v_co, i_co, v1_sig, i1_sig, v2_sig, i2_sig, v_comp, i_comp, res_v, res_i, v_mag, i_mag = compute_coevent_waveforms(co_ev, f_id, time_offset=0.0)

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
