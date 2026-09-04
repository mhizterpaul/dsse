from opendssdirect import dss
import numpy as np
import traceback
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

import src.power_plant.plant as plant
from src.transient.atp_case_builder import (
    ATPCaseBuilder, TransformerSpec, TransformerWinding, ShortCircuitTest,
    ThreePhaseThevenin, TransientEvent, SimulationConfig
)
from src.transient.atp_runner import ATPRunner
from src.transient.atp_parser import ATPOutputReader
from src.transient.events import SingleLineFaultEvent, SingleEquipmentSwitchEvent, EquipmentEquipmentCoEvent


@dataclass
class SimulationResult:
    scenario_id: str
    steady_state_measurements: Dict[str, Any] = field(default_factory=dict)
    time_s: Optional[np.ndarray] = None
    operating_point: Optional[Any] = None


def extract_fault_info(dss_instance: Any, fault_id: str, target_line: str, event_spec: Any) -> str:
    """
    Extracts fault parameters and event specifications.
    """
    if hasattr(event_spec, "event_2") and hasattr(event_spec.event_2, "fault_type"):
        ev_fault = event_spec.event_2
    elif hasattr(event_spec, "event_1") and hasattr(event_spec.event_1, "fault_type"):
        ev_fault = event_spec.event_1
    elif hasattr(event_spec, "fault_type"):
        ev_fault = event_spec
    else:
        return json.dumps({})

    if not hasattr(ev_fault, "fault_type") or not hasattr(ev_fault, "faulted_phases") or not hasattr(ev_fault, "start_time_s") or not hasattr(ev_fault, "duration_s"):
        err_msg = f"Fault event object missing required attributes (fault_type, faulted_phases, start_time_s, duration_s): {ev_fault}"
        print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
        raise ValueError(err_msg)

    fault_info = {
        "fault_id": fault_id,
        "bus": target_line.replace("down_", "f").replace("_", "_node"),
        "target_line": target_line,
        "fault_type": str(ev_fault.fault_type),
        "fault_resistance_ohm": float(getattr(ev_fault, "fault_resistance", getattr(ev_fault, "fault_resistance_ohm", 0.001))),
        "faulted_phases": list(ev_fault.faulted_phases),
        "start_time_s": float(ev_fault.start_time_s),
        "duration_s": float(ev_fault.duration_s)
    }

    return json.dumps(fault_info)


def extract_upstream_thevenin(dss_instance: Any, feeder_idx: int = 1) -> ThreePhaseThevenin:
    """
    Extracts the upstream HV Thévenin source representation (Vth_HV, Zth_HV) looking into the MV feeder.
    """
    head_bus = f"feeder{feeder_idx}_head"
    if not dss_instance.Circuit.SetActiveBus(head_bus):
        raise ValueError(f"Could not set active bus '{head_bus}' in OpenDSS for upstream Thévenin extraction")

    v_mag_angle = dss_instance.Bus.VMagAngle()
    v_hv_complex = (
        float(v_mag_angle[0]) * np.exp(1j * np.radians(float(v_mag_angle[1]))),
        float(v_mag_angle[2]) * np.exp(1j * np.radians(float(v_mag_angle[3]))),
        float(v_mag_angle[4]) * np.exp(1j * np.radians(float(v_mag_angle[5])))
    )

    line_name = f"Line.mv_feeder_{feeder_idx}"
    if dss_instance.Circuit.SetActiveElement(line_name):
        try:
            length_km = float(dss_instance.Properties.Value("length"))
            r1 = float(dss_instance.Properties.Value("r1")) * length_km
            x1 = float(dss_instance.Properties.Value("x1")) * length_km
        except Exception:
            r1 = 0.25 * 4.5
            x1 = 0.35 * 4.5
    else:
        r1 = 0.25 * 4.5
        x1 = 0.35 * 4.5

    z_hv_diag = [r1 + 1j * x1, r1 + 1j * x1, r1 + 1j * x1]
    z_hv_matrix = np.diag(z_hv_diag)

    return ThreePhaseThevenin(
        name=f"upstream_hv_feeder_{feeder_idx}",
        v_th=v_hv_complex,
        z_th=z_hv_matrix,
        v_pre=v_hv_complex,
        i_pre=(0j, 0j, 0j)
    )


def extract_downstream_thevenin(dss_instance: Any, feeder_idx: int = 1) -> ThreePhaseThevenin:
    """
    Extracts the downstream passive LV base network Thévenin representation (Zth_LV)
    looking into the passive LV network from the LV test bus.
    For a passive base load network, Vth_LV = (0j, 0j, 0j) and Zth_LV = Vpre / Ipre.
    """
    sec_bus = f"feeder{feeder_idx}_sec"
    if not dss_instance.Circuit.SetActiveBus(sec_bus):
        raise ValueError(f"Could not set active bus '{sec_bus}' in OpenDSS for downstream Thévenin extraction")

    tx_elem = f"Transformer.trans{feeder_idx}"
    if not dss_instance.Circuit.SetActiveElement(tx_elem):
        raise ValueError(f"Could not set active element '{tx_elem}' in OpenDSS for downstream Thévenin extraction")

    powers = dss_instance.CktElement.Powers()

    p_kw_sec = [abs(float(powers[6])), abs(float(powers[8])), abs(float(powers[10]))]
    q_kvar_sec = [abs(float(powers[7])), abs(float(powers[9])), abs(float(powers[11]))]

    v_mag_angle = dss_instance.Bus.VMagAngle()
    v_pre_complex = (
        float(v_mag_angle[0]) * np.exp(1j * np.radians(float(v_mag_angle[1]))),
        float(v_mag_angle[2]) * np.exp(1j * np.radians(float(v_mag_angle[3]))),
        float(v_mag_angle[4]) * np.exp(1j * np.radians(float(v_mag_angle[5])))
    )

    zth_diag = []
    for idx in range(3):
        v = float(v_mag_angle[2 * idx])
        p = p_kw_sec[idx] * 1000.0
        q = q_kvar_sec[idx] * 1000.0
        s_sq = p**2 + q**2
        if s_sq > 0:
            r = (v**2 * p) / s_sq
            x = (v**2 * q) / s_sq
        else:
            r = 100.0
            x = 10.0
        z_val = r + 1j * x
        zth_diag.append(z_val)

    zth_matrix = np.diag(zth_diag)

    # For passive downstream LV load networks, open-circuit Vth is zero
    return ThreePhaseThevenin(
        name=f"base_lv_network_feeder{feeder_idx}",
        v_th=(0j, 0j, 0j),
        z_th=zth_matrix,
        v_pre=v_pre_complex,
        i_pre=(0j, 0j, 0j)
    )


class CoSimulationRunner:
    """
    Co-Simulation Orchestrator that energizes the imported plant from src.power_plant,
    handles steady state operation and reduced-order EMT transient evaluation via BCTRAN.
    """
    def __init__(self):
        self.dss = dss
        self.atp_builder = ATPCaseBuilder()
        self.plant_data = None
        self._op_cache = {}
        self._atp_response_cache = {}

    def initialize_plant_session(
        self,
        use_baseline_feeder: bool = True,
        seed: int = 42,
        verbose: bool = False
    ) -> dict:
        """
        Initializes a single constant OpenDSS DSS instance for a dataset generation loop.
        """
        try:
            if use_baseline_feeder:
                self.plant_data = plant.build_single_lv_network_composition(
                    dss=self.dss,
                    seed=seed,
                    verbose=verbose
                )
            else:
                self.plant_data = plant.build_three_lv_networks_composition(
                    dss=self.dss,
                    use_baseline_transformers=use_baseline_feeder,
                    seed=seed,
                    verbose=verbose
                )
            if verbose:
                print("INFO: Plant session initialized successfully.")
            return self.plant_data
        except Exception as e:
            print(f"ERROR: Failed to initialize plant session: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Plant session initialization failure: {e}") from e

    def measure_transients(
        self,
        op: Any,
        event: Any,
        scenario_id: str,
        upstream: Optional[ThreePhaseThevenin] = None,
        downstream: Optional[ThreePhaseThevenin] = None,
        feeder_idx: int = 1,
        use_baseline_feeder: bool = True
    ) -> tuple[np.ndarray, dict, dict, dict]:
        """
        Exclusively executes ATP transient simulation cases using two-port Thévenin reductions
        (upstream HV source and downstream LV base-network equivalent) alongside BCTRAN models.
        Target test branches are constructed entirely inside ATPCaseBuilder from the event object.
        """
        if event is None:
            err_msg = f"Cannot measure transients for None event in scenario {scenario_id}"
            print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
            raise ValueError(err_msg)

        if not op:
            err_msg = f"Operating point must be provided for transient measurement in scenario {scenario_id}"
            print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
            raise ValueError(err_msg)

        from src.power_plant.lv_transformers import get_distribution_transformer_spec
        tx_spec = get_distribution_transformer_spec(feeder_idx, use_baseline=use_baseline_feeder)

        target_tx = f"trans{feeder_idx}"
        feeder_id = f"feeder_{feeder_idx}"

        if hasattr(event, "event_1") and hasattr(event, "event_2") and event.event_1 is not None and event.event_2 is not None:
            t_off = getattr(event, "time_offset_s", 0.0)
            ev1_t = getattr(event.event_1, "equipment_type", getattr(event.event_1, "fault_type", "event1"))
            ev2_t = getattr(event.event_2, "equipment_type", getattr(event.event_2, "fault_type", "event2"))
            ev_key = f"{ev1_t}_{ev2_t}_coevent_{t_off:.4f}s"
        elif hasattr(event, "equipment_type") and event.equipment_type:
            ev_key = f"{event.equipment_type}_switch"
        elif hasattr(event, "fault_type") and event.fault_type:
            f_phases = event.faulted_phases
            f_res = getattr(event, "fault_resistance", getattr(event, "fault_resistance_ohm", 0.001))
            ev_key = f"{event.fault_type}_{'-'.join(map(str, f_phases))}_{f_res}"
        elif hasattr(event, "event_type"):
            ev_key = f"event_{event.event_type}"
        else:
            ev_key = f"event_{scenario_id}"

        v_tuple = op.phase_voltages_v[target_tx]
        a_tuple = op.phase_angles_deg[target_tx]
        freq = op.frequency_hz

        atp_cache_key = (ev_key, feeder_id, target_tx, v_tuple, a_tuple, freq)
        if atp_cache_key in self._atp_response_cache:
            return self._atp_response_cache[atp_cache_key]

        atp_case_path = f"src/simulation/atp_cases/case_{ev_key}_{scenario_id}_{os.getpid()}.ATP"

        try:
            target_tx_key = f"trans{feeder_idx}"
            kvas = tx_spec["kvas"]
            kvs = tx_spec["kvs"]
            r_pct = float(tx_spec["r_pct"])
            xhl_pct = float(tx_spec["xhl_pct"])
            r0_pct = float(tx_spec.get("r0_pct", r_pct))
            x0_pct = float(tx_spec.get("x0_pct", xhl_pct))
            noloadloss_pct = float(tx_spec.get("noloadloss_pct", 0.1))
            imag_pct = float(tx_spec.get("imag_pct", 0.8))

            z0_pu = float(np.sqrt((r0_pct / 100.0) ** 2 + (x0_pct / 100.0) ** 2))
            losses_zero_kw = (r0_pct / 100.0) * float(kvas[0])

            transformer = TransformerSpec(
                name=str(tx_spec["name"]),
                frequency_hz=float(op.frequency_hz),
                windings=[
                    TransformerWinding("HV", float(kvs[0]), float(kvas[0]) / 1000.0, "Y", 0.0),
                    TransformerWinding("LV", float(kvs[1]), float(kvas[1]) / 1000.0, "Y", 0.0)
                ],
                short_circuit_tests=[
                    ShortCircuitTest(1, 2, z_pos_pu=float(np.sqrt((r_pct / 100.0) ** 2 + (xhl_pct / 100.0) ** 2)), losses_pos_kw=(r_pct / 100.0) * float(kvas[0]), z_zero_pu=z0_pu, losses_zero_kw=losses_zero_kw)
                ],
                excitation_current_percent=imag_pct,
                excitation_loss_kw=(noloadloss_pct / 100.0) * float(kvas[0]),
                vector_group="Yy0"
            )

            if upstream is None:
                upstream = extract_upstream_thevenin(self.dss, feeder_idx=feeder_idx)

            if downstream is None:
                downstream = extract_downstream_thevenin(self.dss, feeder_idx=feeder_idx)

            if hasattr(event, "event_1") and hasattr(event, "event_2") and event.event_1 is not None and event.event_2 is not None:
                ev_class = "co_event"
                start_s = float(getattr(event.event_1, "start_time_s", 0.02))
                dur_s = float(getattr(event.event_1, "duration_s", 0.5))
            else:
                ev_class = str(getattr(event, "event_class", "equipment_switch"))
                start_s = float(getattr(event, "start_time_s", 0.02))
                dur_s = float(getattr(event, "duration_s", 0.5))

            f_type = getattr(event, "fault_type", None)
            f_phases = getattr(event, "faulted_phases", (0,))
            f_res = float(getattr(event, "fault_resistance", getattr(event, "fault_resistance_ohm", 0.001)))

            transient_ev = TransientEvent(
                event_class=ev_class,
                start_time_s=start_s,
                duration_s=dur_s,
                location=f"trans{feeder_idx}",
                fault_type=f_type,
                faulted_phases=tuple(f_phases),
                fault_resistance_ohm=f_res,
                equipment_type=getattr(event, "equipment_type", None),
                event_1=getattr(event, "event_1", None),
                event_2=getattr(event, "event_2", None)
            )

            sim_config = SimulationConfig(t_start_s=0.0, t_stop_s=start_s + dur_s + 0.05, time_step_s=1e-4)

            self.atp_builder.build_explicit(
                transformer=transformer,
                upstream=upstream,
                downstream=downstream,
                event=transient_ev,
                simulation=sim_config,
                output_path=atp_case_path,
                scenario_id=scenario_id
            )

            atp_result = ATPRunner().run(atp_case_path)
            emt_waveforms = ATPOutputReader().read(atp_result, event, transformer_id=target_tx)

            if emt_waveforms is None or emt_waveforms.time_s is None or len(emt_waveforms.time_s) == 0:
                raise ValueError(f"ATP simulation returned empty waveforms for scenario {scenario_id}")

            return emt_waveforms.time_s, emt_waveforms.voltages, emt_waveforms.currents, emt_waveforms.event_metadata

        except Exception as e:
            lis_path = Path(atp_case_path).with_suffix(".lis")
            lis_debug_content = ""
            if lis_path.exists():
                try:
                    lis_debug_content = f"\n--- ATP LIS Log Output ---\n{lis_path.read_text(errors='replace')[-2000:]}"
                except Exception:
                    pass
            err_msg = f"Failed ATP transient measurement for scenario '{scenario_id}': {e}{lis_debug_content}"
            print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
            raise ValueError(err_msg) from e

    def run_steady_state_simulation(
        self,
        use_baseline_feeder: bool = False,
        scenario_id: str = "steady_5min_run",
        seed: int = 42,
        reinitialize_plant: bool = True,
        verbose: bool = False
    ) -> SimulationResult:
        """
        Executes a pure OpenDSS steady-state power flow simulation for Dataset 1.
        """
        if reinitialize_plant or self.plant_data is None:
            plant_data = self.initialize_plant_session(
                use_baseline_feeder=use_baseline_feeder,
                seed=seed,
                verbose=verbose
            )
        else:
            plant_data = self.plant_data

        self.dss.run_command("disable Fault.*")
        self.dss.run_command("set stepsize=1s")
        self.dss.run_command("set number=300")
        self.dss.run_command("set mode=daily")
        self.dss.run_command("solve")

        op = plant.solve_operating_point(self.dss)

        return SimulationResult(
            scenario_id=scenario_id,
            operating_point=op
        )

    def run_transient_simulation(
        self,
        events: List[Any],
        use_baseline_feeder: bool = True,
        seed: int = 42,
        reinitialize_plant: bool = True,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Executes ATP transient simulations evaluated directly for a list of co-events.
        Uses ProcessPoolExecutor with max_workers=6 to process pairs in parallel.
        """
        from concurrent.futures import ProcessPoolExecutor

        tasks = [
            (ev, use_baseline_feeder, seed + idx, idx)
            for idx, ev in enumerate(events)
        ]

        with ProcessPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(_simulate_single_coevent_worker, tasks))

        return results


def _simulate_single_coevent_worker(args_tuple: tuple) -> Dict[str, Any]:
    """
    Worker function executed in parallel across process workers.
    Solves pre-event operating state ONLY in OpenDSS, extracts upstream and downstream Thévenin equivalents,
    and runs ATP transient simulation for Event 1, Event 2, and Joint Co-Event without adding test elements to OpenDSS.
    """
    co_ev, use_baseline_feeder, seed, task_idx = args_tuple

    runner = CoSimulationRunner()
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)

    ev1 = co_ev.event_1
    ev2 = co_ev.event_2

    if use_baseline_feeder:
        feeder_idx = 1
    else:
        if hasattr(co_ev, "gt_feeder_id"):
            raw_fid = co_ev.gt_feeder_id
            feeder_idx = int(raw_fid[7:]) if isinstance(raw_fid, str) and raw_fid.startswith("feeder_") else int(raw_fid)
        else:
            feeder_idx = (task_idx % 3) + 1

    target_line = f"feeder{feeder_idx}_head"

    # --- Step 1: Solve pre-event operating state ONLY in OpenDSS ---
    runner.dss.run_command("disable Fault.*")
    op0 = plant.solve_operating_point(runner.dss)

    # --- Step 2: Extract two-port Thévenin reductions (Upstream HV & Downstream LV) ---
    upstream_th = extract_upstream_thevenin(runner.dss, feeder_idx=feeder_idx)
    downstream_th = extract_downstream_thevenin(runner.dss, feeder_idx=feeder_idx)

    # --- Step 3: Run ATP simulations directly passing events ---
    t_joint, v_joint_dict, i_joint_dict, _ = runner.measure_transients(
        op=op0,
        event=co_ev,
        scenario_id=f"p{os.getpid()}_{task_idx}_joint",
        upstream=upstream_th,
        downstream=downstream_th,
        feeder_idx=feeder_idx,
        use_baseline_feeder=use_baseline_feeder
    )

    t1, v1_dict, i1_dict, _ = runner.measure_transients(
        op=op0,
        event=ev1,
        scenario_id=f"p{os.getpid()}_{task_idx}_ev1",
        upstream=upstream_th,
        downstream=downstream_th,
        feeder_idx=feeder_idx,
        use_baseline_feeder=use_baseline_feeder
    )

    t2, v2_dict, i2_dict, _ = runner.measure_transients(
        op=op0,
        event=ev2,
        scenario_id=f"p{os.getpid()}_{task_idx}_ev2",
        upstream=upstream_th,
        downstream=downstream_th,
        feeder_idx=feeder_idx,
        use_baseline_feeder=use_baseline_feeder
    )

    tx_unit_id = f"trans{feeder_idx}_lv_boundary"

    for d_name, d_val in [
        ("v1_dict", v1_dict),
        ("i1_dict", i1_dict),
        ("v2_dict", v2_dict),
        ("i2_dict", i2_dict),
        ("v_joint_dict", v_joint_dict),
        ("i_joint_dict", i_joint_dict),
    ]:
        if tx_unit_id not in d_val:
            err_msg = f"Missing waveform key '{tx_unit_id}' in {d_name} for co-event scenario index {task_idx}"
            print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
            raise ValueError(err_msg)

    v1 = v1_dict[tx_unit_id]
    i1 = i1_dict[tx_unit_id]
    v2 = v2_dict[tx_unit_id]
    i2 = i2_dict[tx_unit_id]
    v_joint = v_joint_dict[tx_unit_id]
    i_joint = i_joint_dict[tx_unit_id]

    active_fault_id = "F_joint" if (hasattr(co_ev, "fault_type") or hasattr(ev1, "fault_type") or hasattr(ev2, "fault_type")) else None
    fault_info_dict = json.loads(extract_fault_info(runner.dss, active_fault_id, target_line, co_ev)) if active_fault_id else {}

    return {
        "co_ev": co_ev,
        "time_s": t_joint,
        "v1": v1,
        "i1": i1,
        "v2": v2,
        "i2": i2,
        "v_joint": v_joint,
        "i_joint": i_joint,
        "fault_info": json.dumps(fault_info_dict),
        "gt_feeder_id": f"feeder_{feeder_idx}"
    }
