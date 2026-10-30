from opendssdirect import dss
import numpy as np
import traceback
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

import scipy.signal
import scipy.ndimage
import src.power_plant.plant as plant
from src.transient.models import (
    ThreePhaseThevenin,
    TestBranch,
    SimulationConfig,
)
from src.transient.opendss_reducer import OpenDSSReducer
from src.transient.bctran_generator import BCTRANGenerator
from src.transient.atp_case_builder import ATPCaseBuilder
from src.transient.atp_runner import ATPRunner
from src.transient.atp_parser import ATPOutputReader, _extract_fundamental_phase
from src.transient.events import (
    SingleLineFaultEvent,
    SingleEquipmentSwitchEvent,
    EquipmentEquipmentCoEvent,
    EquipmentLineFaultCoEvent,
    NoLoadEvent,
)
from src.power_plant.lv_transformers import build_transformer_spec


@dataclass
class SimulationResult:
    scenario_id: str
    steady_state_measurements: Dict[str, Any] = field(default_factory=dict)
    time_s: Optional[np.ndarray] = None
    operating_point: Optional[Any] = None


def compute_atp_phase_angle(
    waveform: np.ndarray,
    time_s: Optional[np.ndarray] = None,
    frequency: float = 50.0,
    op_reference_deg: Optional[Any] = 0.0,
    window_cycles: float = 0.5,
    amplitude_threshold: float = 1e-4,
    return_details: bool = False,
) -> Any:
    """
    Derives instantaneous fundamental phase angles (radians) directly from raw ATP-EMTP transient simulation waveforms
    referenced against OpenDSS operating point phasors. Delegates to _extract_fundamental_phase in atp_parser.py.
    """
    if time_s is None:
        time_s = np.arange(len(waveform)) * 1e-4

    n_cols = waveform.shape[1] if waveform.ndim > 1 else 1
    if isinstance(op_reference_deg, (list, tuple, np.ndarray)):
        ref_rad = np.radians(op_reference_deg)
    else:
        ref_deg_val = float(op_reference_deg) if op_reference_deg is not None else 0.0
        ref_rad = np.full(n_cols, np.radians(ref_deg_val))

    res = _extract_fundamental_phase(
        time_s=time_s,
        waveform=waveform,
        frequency_hz=frequency,
        reference_angles_rad=ref_rad,
        window_cycles=window_cycles,
        amplitude_threshold_ratio=amplitude_threshold,
    )

    if return_details:
        return res
    return res["phase_abs"]


def extract_transformer_phasors_from_dss(dss_instance: Any, feeder_idx: int) -> dict:
    """
    Extracts OpenDSS LV terminal complex voltage and current phasors for the active transformer boundary.
    """
    tx_unit_id = f"trans{feeder_idx}_lv_boundary"
    tx_name = f"Transformer.trans{feeder_idx}"
    sec_bus = f"feeder{feeder_idx}_sec"

    v_phasors = np.zeros(3, dtype=complex)
    i_phasors = np.zeros(3, dtype=complex)

    if dss_instance.Circuit.SetActiveElement(tx_name):
        c_raw = dss_instance.CktElement.Currents()
        if len(c_raw) >= 12:
            i_phasors = np.array([
                complex(c_raw[6], c_raw[7]),
                complex(c_raw[8], c_raw[9]),
                complex(c_raw[10], c_raw[11])
            ])

    if dss_instance.Circuit.SetActiveBus(sec_bus):
        v_comp = dss_instance.Bus.Voltages()
        if len(v_comp) >= 6:
            v_phasors = np.array([
                complex(v_comp[0], v_comp[1]),
                complex(v_comp[2], v_comp[3]),
                complex(v_comp[4], v_comp[5])
            ])

    return {
        tx_unit_id: {
            "voltage_abc": v_phasors,
            "current_abc": i_phasors
        }
    }


def extract_fault_info(dss_instance: Any, fault_id: str, target_line: str, event_spec: Any) -> str:
    """
    Extracts fault parameters (fault_type, fault_resistance_ohm, faulted_phases, line parameters,
    phase and neutral line currents) directly from event specifications and active OpenDSS line elements.
    Raises ValueError if required fault parameters or line elements are missing or invalid.
    """
    ev_fault = None
    if hasattr(event_spec, "event_2") and hasattr(event_spec.event_2, "fault_type"):
        ev_fault = event_spec.event_2
    elif hasattr(event_spec, "event_1") and hasattr(event_spec.event_1, "fault_type"):
        ev_fault = event_spec.event_1
    elif hasattr(event_spec, "fault_type"):
        ev_fault = event_spec
    else:
        return json.dumps({})

    if not hasattr(ev_fault, "fault_resistance"):
        raise ValueError(f"Fault event spec missing 'fault_resistance': {ev_fault}")
    if not hasattr(ev_fault, "fault_type"):
        raise ValueError(f"Fault event spec missing 'fault_type': {ev_fault}")
    if not hasattr(ev_fault, "faulted_phases"):
        raise ValueError(f"Fault event spec missing 'faulted_phases': {ev_fault}")
    if not hasattr(ev_fault, "start_time_s"):
        raise ValueError(f"Fault event spec missing 'start_time_s': {ev_fault}")
    if not hasattr(ev_fault, "duration_s"):
        raise ValueError(f"Fault event spec missing 'duration_s': {ev_fault}")

    fault_r = float(ev_fault.fault_resistance)
    fault_type = str(ev_fault.fault_type)
    faulted_phases = list(ev_fault.faulted_phases)
    start_time_s = float(ev_fault.start_time_s)
    duration_s = float(ev_fault.duration_s)

    # Query Line element parameters and conductor currents
    line_elem = f"Line.{target_line}"
    if dss_instance.Circuit.SetActiveElement(line_elem):
        line_r1 = float(dss_instance.Properties.Value("r1"))
        line_x1 = float(dss_instance.Properties.Value("x1"))
        c_raw = dss_instance.CktElement.Currents()
        f_currents = [float(c) for c in c_raw]
    else:
        raise ValueError(f"Target line '{line_elem}' could not be activated in OpenDSS")

    # Query Fault element if present
    fault_elem = f"Fault.{fault_id}"
    if dss_instance.Circuit.SetActiveElement(fault_elem):
        c_fault_raw = dss_instance.CktElement.Currents()
        if len(c_fault_raw) > 0:
            f_currents = [float(c) for c in c_fault_raw]

    # Compute neutral line current (conductor 4 or sum of 3 phase currents)
    if len(f_currents) >= 8:
        i_n = complex(f_currents[6], f_currents[7])
    elif len(f_currents) >= 6:
        i_a = complex(f_currents[0], f_currents[1])
        i_b = complex(f_currents[2], f_currents[3])
        i_c = complex(f_currents[4], f_currents[5])
        i_n = i_a + i_b + i_c
    else:
        i_n = complex(0.0, 0.0)

    fault_info = {
        "fault_id": fault_id,
        "bus": target_line,
        "target_line": target_line,
        "fault_type": fault_type,
        "fault_resistance_ohm": fault_r,
        "faulted_phases": faulted_phases,
        "fault_currents": f_currents,
        "neutral_current_a": float(abs(i_n)),
        "line_r1_ohm": line_r1,
        "line_x1_ohm": line_x1,
        "start_time_s": start_time_s,
        "duration_s": duration_s,
    }

    return json.dumps(fault_info)


class CoSimulationRunner:
    """
    Co-Simulation Orchestrator that energizes the imported plant from src.power_plant,
    handles 2 network cases (single LV network composition vs 3 LV networks composition),
    performs OpenDSS network reduction to multi-phase Thévenin equivalents,
    uses BCTRANGenerator for BCTRAN model punch matrix,
    and executes focused ATP EMT models for equipment and fault test branches.
    """

    def __init__(self):
        self.dss = dss
        self.dss_reducer = OpenDSSReducer()
        self.bctran_generator = BCTRANGenerator()
        self.atp_builder = ATPCaseBuilder()
        self.plant_data = None
        self._op_cache = {}
        self._atp_response_cache = {}
        self._bctran_cache = {}

    def initialize_plant_session(
        self,
        use_baseline_feeder: bool = True,
        seed: int = 42,
        verbose: bool = False,
    ) -> dict:
        """
        Initializes a single constant OpenDSS DSS instance for a dataset generation loop.
        """
        try:
            if use_baseline_feeder:
                self.plant_data = plant.build_single_lv_network_composition(
                    dss=self.dss, seed=seed, verbose=verbose
                )
            else:
                self.plant_data = plant.build_three_lv_networks_composition(
                    dss=self.dss,
                    use_baseline_transformers=use_baseline_feeder,
                    seed=seed,
                    verbose=verbose,
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
        feeder_idx: int = 1,
        use_baseline_feeder: bool = True,
        t_stop_override: Optional[float] = None,
    ) -> tuple[np.ndarray, dict, dict, dict]:
        """
        Executes ATP transient simulation cases using reduced OpenDSS Thévenin equivalents,
        BCTRAN transformer punch matrix models, and explicit test branches.
        """
        if event is None:
            err_msg = f"Cannot measure transients for None event in scenario {scenario_id}"
            print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
            raise ValueError(err_msg)

        if not op:
            err_msg = f"Operating point must be provided for transient measurement in scenario {scenario_id}"
            print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
            raise ValueError(err_msg)

        target_tx = f"trans{feeder_idx}"
        feeder_id = f"feeder_{feeder_idx}"

        if hasattr(event, "event_1") and hasattr(event, "event_2"):
            t_off = event.time_offset_s
            ev_key = f"{event.event_1.event_type}_{event.event_2.event_type}_coevent_{t_off:.4f}s"
        elif hasattr(event, "equipment_type"):
            ev_key = f"{event.equipment_type}_switch"
        elif hasattr(event, "fault_type"):
            f_phases = event.faulted_phases
            f_res = event.fault_resistance
            ev_key = f"{event.fault_type}_{'-'.join(map(str, f_phases))}_{f_res}"
        elif hasattr(event, "event_type"):
            ev_key = f"event_{event.event_type}"
        else:
            err_msg = f"Event object missing recognizable event type attributes: {event}"
            print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
            raise ValueError(err_msg)

        freq = float(op.frequency_hz)
        atp_case_path = f"src/simulation/atp_cases/case_{ev_key}_{scenario_id}_{os.getpid()}.ATP"

        try:
            # 1. Resolve event ports & perform OpenDSS pre-event network reduction
            event_ports = self.dss_reducer.resolve_event_ports(
                self.dss, event=event, feeder_idx=feeder_idx
            )

            upstream_th = self.dss_reducer.reduce_upstream_thevenin(
                self.dss, tx_name=event_ports["tx_name"]
            )
            downstream_th = self.dss_reducer.reduce_downstream_thevenin(
                self.dss, test_bus=event_ports["test_bus"], tx_name=event_ports["tx_name"]
            )

            # 2. Build TransformerSpec domain object and retrieve/generate BCTRAN punch matrix
            tx_spec = build_transformer_spec(
                feeder_idx=feeder_idx, use_baseline=use_baseline_feeder, frequency_hz=freq
            )

            tx_cache_key = (tx_spec.name, float(freq))
            if tx_cache_key not in self._bctran_cache:
                self._bctran_cache[tx_cache_key] = self.bctran_generator.generate(tx_spec)
            bctran_punch = self._bctran_cache[tx_cache_key]

            # 3. Convert event into explicit TestBranch list
            if hasattr(event, "to_test_branches"):
                test_branches = event.to_test_branches(freq)
            else:
                test_branches = []

            # Determine simulation duration based on max event end time or override
            if t_stop_override is not None:
                t_stop = float(t_stop_override)
            elif test_branches:
                max_end = max(b.end_time_s for b in test_branches)
                t_stop = max(max_end + 0.05, 0.15)
            else:
                t_stop = 0.15

            sim_config = SimulationConfig(t_start_s=0.0, t_stop_s=t_stop, time_step_s=1e-4)

            # 4. Build explicit ATP case
            self.atp_builder.build_explicit(
                transformer=tx_spec,
                upstream=upstream_th,
                downstream=downstream_th,
                events=test_branches,
                simulation=sim_config,
                bctran_punch=bctran_punch,
                scenario_id=scenario_id,
                output_path=atp_case_path,
            )

            # 5. Execute ATP and parse waveforms
            atp_result = ATPRunner().run(atp_case_path)
            emt_waveforms = ATPOutputReader().read(
                atp_result, event, transformer_id=f"{target_tx}_lv_boundary"
            )

            if emt_waveforms is None or emt_waveforms.time_s is None or len(emt_waveforms.time_s) == 0:
                raise ValueError(
                    f"ATP simulation returned empty waveforms for scenario {scenario_id}"
                )

            return (
                emt_waveforms.time_s,
                emt_waveforms.voltages,
                emt_waveforms.currents,
                emt_waveforms.event_metadata,
            )

        except Exception as e:
            lis_path = Path(atp_case_path).with_suffix(".lis")
            lis_debug_content = ""
            if lis_path.exists():
                try:
                    lis_debug_content = (
                        f"\n--- ATP LIS Log Output ---\n{lis_path.read_text(errors='replace')[-2000:]}"
                    )
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
        verbose: bool = False,
    ) -> SimulationResult:
        """
        Executes pure OpenDSS steady-state power flow simulation for Dataset 1.
        """
        if reinitialize_plant or self.plant_data is None:
            plant_data = self.initialize_plant_session(
                use_baseline_feeder=use_baseline_feeder, seed=seed, verbose=verbose
            )
        else:
            plant_data = self.plant_data

        self.dss.run_command("disable Fault.*")
        self.dss.run_command("set stepsize=1s")
        self.dss.run_command("set number=300")
        self.dss.run_command("set mode=daily")
        self.dss.run_command("solve")

        op = plant.solve_operating_point(self.dss)

        return SimulationResult(scenario_id=scenario_id, operating_point=op)

    def run_transient_simulation(
        self,
        events: List[Any],
        use_baseline_feeder: bool = True,
        seed: int = 42,
        reinitialize_plant: bool = True,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Executes ATP transient simulations evaluated directly for a list of co-events.
        Uses ProcessPoolExecutor with max_workers=6 to process pairs in parallel.
        """
        from concurrent.futures import ProcessPoolExecutor

        bctran_cache = dict(self._bctran_cache)
        for f_idx in (1, 2, 3):
            tx_spec = build_transformer_spec(
                feeder_idx=f_idx, use_baseline=use_baseline_feeder, frequency_hz=50.0
            )
            tx_key = (tx_spec.name, 50.0)
            if tx_key not in bctran_cache:
                bctran_cache[tx_key] = self.bctran_generator.generate(tx_spec)
        self._bctran_cache.update(bctran_cache)

        tasks = [
            (ev, use_baseline_feeder, seed + idx, idx, bctran_cache)
            for idx, ev in enumerate(events)
        ]

        with ProcessPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(_simulate_single_coevent_worker, tasks))

        return results


def _simulate_single_coevent_worker(args_tuple: tuple) -> Dict[str, Any]:
    """
    Worker function executed in parallel across 6 process workers.
    Reuses process-level CoSimulationRunner and its OpenDSS instance.
    OpenDSS solves pre-event steady-state operating points of the base network.
    Test loads and faults are instantiated exclusively in ATP.
    """
    co_ev, use_baseline_feeder, seed, task_idx, bctran_cache = args_tuple

    runner = CoSimulationRunner()
    runner._bctran_cache.update(bctran_cache)
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)

    ev1 = co_ev.event_1
    ev2 = co_ev.event_2

    # Determine feeder connected to event
    if use_baseline_feeder:
        feeder_idx = 1
    else:
        if hasattr(co_ev, "gt_feeder_id"):
            raw_fid = co_ev.gt_feeder_id
            feeder_idx = (
                int(raw_fid[7:])
                if isinstance(raw_fid, str) and raw_fid.startswith("feeder_")
                else int(raw_fid)
            )
        else:
            feeder_idx = (task_idx % 3) + 1

    target_bus = f"feeder{feeder_idx}_sec"
    target_line = f"mv_feeder_{feeder_idx}"

    # Calculate uniform t_stop based on co-event max branch duration
    branches_co = co_ev.to_test_branches(50.0)
    co_max_end = max(b.end_time_s for b in branches_co) if branches_co else 0.15
    uniform_t_stop = max(co_max_end + 0.05, 0.15)

    tx_unit_id = f"trans{feeder_idx}_lv_boundary"

    # --- STEP 0: Pre-event base network operating point & No-Load Baseline simulation ---
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
    runner.dss.run_command("disable Fault.*")
    op_base = plant.solve_operating_point(runner.dss)
    op_base.transformer_phasors = extract_transformer_phasors_from_dss(runner.dss, feeder_idx)

    no_load_ev = NoLoadEvent(start_time_s=0.02, duration_s=0.10)
    t_base, v_base_dict, i_base_dict, _ = runner.measure_transients(
        op=op_base,
        event=no_load_ev,
        scenario_id=f"p{os.getpid()}_{task_idx}_base",
        feeder_idx=feeder_idx,
        use_baseline_feeder=use_baseline_feeder,
        t_stop_override=uniform_t_stop,
    )

    # --- STEP 1: Pre-event base network operating point & joint co-event ATP simulation ---
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
    runner.dss.run_command("disable Fault.*")
    op_joint = plant.solve_operating_point(runner.dss)
    op_joint.transformer_phasors = extract_transformer_phasors_from_dss(runner.dss, feeder_idx)

    fault_info_json = extract_fault_info(runner.dss, "joint_fault", target_line, co_ev)

    t_joint, v_joint_dict, i_joint_dict, _ = runner.measure_transients(
        op=op_joint,
        event=co_ev,
        scenario_id=f"p{os.getpid()}_{task_idx}_joint",
        feeder_idx=feeder_idx,
        use_baseline_feeder=use_baseline_feeder,
        t_stop_override=uniform_t_stop,
    )

    # --- STEP 2: Pre-event base network operating point & Event 1 ATP simulation ---
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
    runner.dss.run_command("disable Fault.*")
    op_1 = plant.solve_operating_point(runner.dss)
    op_1.transformer_phasors = extract_transformer_phasors_from_dss(runner.dss, feeder_idx)

    t1, v1_dict, i1_dict, _ = runner.measure_transients(
        op=op_1,
        event=ev1,
        scenario_id=f"p{os.getpid()}_{task_idx}_ev1",
        feeder_idx=feeder_idx,
        use_baseline_feeder=use_baseline_feeder,
        t_stop_override=uniform_t_stop,
    )

    # --- STEP 3: Pre-event base network operating point & Event 2 ATP simulation ---
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
    runner.dss.run_command("disable Fault.*")
    op_2 = plant.solve_operating_point(runner.dss)
    op_2.transformer_phasors = extract_transformer_phasors_from_dss(runner.dss, feeder_idx)
    t2, v2_dict, i2_dict, _ = runner.measure_transients(
        op=op_2,
        event=ev2,
        scenario_id=f"p{os.getpid()}_{task_idx}_ev2",
        feeder_idx=feeder_idx,
        use_baseline_feeder=use_baseline_feeder,
        t_stop_override=uniform_t_stop,
    )

    for d_name, d_val in [
        ("v_base_dict", v_base_dict),
        ("i_base_dict", i_base_dict),
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

    v_base = v_base_dict[tx_unit_id]
    i_base = i_base_dict[tx_unit_id]

    v1_raw = v1_dict[tx_unit_id]
    i1_raw = i1_dict[tx_unit_id]
    v2_raw = v2_dict[tx_unit_id]
    i2_raw = i2_dict[tx_unit_id]
    v_joint_raw = v_joint_dict[tx_unit_id]
    i_joint_raw = i_joint_dict[tx_unit_id]

    # Remove no-load baseline contribution from single events and joint events
    v1 = v1_raw - v_base
    i1 = i1_raw - i_base
    v2 = v2_raw - v_base
    i2 = i2_raw - i_base
    v_joint = v_joint_raw - v_base
    i_joint = i_joint_raw - i_base

    # Extract OpenDSS voltage and current complex reference phasors from OperatingPoint
    def _extract_op_ref(op_obj, tx_id):
        if hasattr(op_obj, "transformer_phasors") and op_obj.transformer_phasors and tx_id in op_obj.transformer_phasors:
            p_dict = op_obj.transformer_phasors[tx_id]
            v_c = p_dict["voltage_abc"]
            i_c = p_dict["current_abc"]
            return np.angle(v_c), np.abs(v_c), np.angle(i_c), np.abs(i_c)
        def_ang = np.radians([0.0, -120.0, 120.0])
        def_v_mag = np.array([230.0 * np.sqrt(2)] * 3)
        def_i_mag = np.array([10.0] * 3)
        return def_ang, def_v_mag, def_ang, def_i_mag

    v1_ref_ang, v1_ref_mag, i1_ref_ang, i1_ref_mag = _extract_op_ref(op_1, tx_unit_id)
    v2_ref_ang, v2_ref_mag, i2_ref_ang, i2_ref_mag = _extract_op_ref(op_2, tx_unit_id)
    vj_ref_ang, vj_ref_mag, ij_ref_ang, ij_ref_mag = _extract_op_ref(op_joint, tx_unit_id)

    reader = ATPOutputReader()

    wf_1 = EMTWaveforms(t1, {tx_unit_id: v1_raw}, {tx_unit_id: i1_raw}, {}, float(op_1.frequency_hz))
    wf_2 = EMTWaveforms(t2, {tx_unit_id: v2_raw}, {tx_unit_id: i2_raw}, {}, float(op_2.frequency_hz))
    wf_joint = EMTWaveforms(t_joint, {tx_unit_id: v_joint_raw}, {tx_unit_id: i_joint_raw}, {}, float(op_joint.frequency_hz))

    pt_1 = reader.extract_phase_trajectories(
        wf_1, tx_unit_id,
        voltage_reference_angles_rad=v1_ref_ang,
        current_reference_angles_rad=i1_ref_ang,
        voltage_reference_magnitudes=v1_ref_mag,
        current_reference_magnitudes=i1_ref_mag,
    )
    pt_2 = reader.extract_phase_trajectories(
        wf_2, tx_unit_id,
        voltage_reference_angles_rad=v2_ref_ang,
        current_reference_angles_rad=i2_ref_ang,
        voltage_reference_magnitudes=v2_ref_mag,
        current_reference_magnitudes=i2_ref_mag,
    )
    pt_joint = reader.extract_phase_trajectories(
        wf_joint, tx_unit_id,
        voltage_reference_angles_rad=vj_ref_ang,
        current_reference_angles_rad=ij_ref_ang,
        voltage_reference_magnitudes=vj_ref_mag,
        current_reference_magnitudes=ij_ref_mag,
    )

    v1_phase = pt_1["voltage"]["phase_abs"]
    i1_phase = pt_1["current"]["phase_abs"]
    v2_phase = pt_2["voltage"]["phase_abs"]
    i2_phase = pt_2["current"]["phase_abs"]
    v_joint_phase = pt_joint["voltage"]["phase_abs"]
    i_joint_phase = pt_joint["current"]["phase_abs"]

    return {
        "co_ev": co_ev,
        "time_s": t_joint,
        "v1": v1,
        "i1": i1,
        "v2": v2,
        "i2": i2,
        "v_joint": v_joint,
        "i_joint": i_joint,
        "v1_phase": v1_phase,
        "i1_phase": i1_phase,
        "v2_phase": v2_phase,
        "i2_phase": i2_phase,
        "v_joint_phase": v_joint_phase,
        "i_joint_phase": i_joint_phase,
        "fault_info": fault_info_json,
        "gt_feeder_id": f"feeder_{feeder_idx}",
    }
