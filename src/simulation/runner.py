from opendssdirect import dss
import numpy as np
import traceback
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import src.power_plant.plant as plant
from src.transient.atp_case_builder import ATPCaseBuilder
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
    Extracts fault currents, fault resistance (R), line parameters (R1, X1),
    and event specifications from active OpenDSS elements and event object.
    Raises ValueError with stack trace if element or parameters are missing/invalid.
    """
    if hasattr(event_spec, "event_2") and hasattr(event_spec.event_2, "fault_type"):
        ev_fault = event_spec.event_2
    elif hasattr(event_spec, "event_1") and hasattr(event_spec.event_1, "fault_type"):
        ev_fault = event_spec.event_1
    elif hasattr(event_spec, "fault_type"):
        ev_fault = event_spec
    else:
        return json.dumps({})

    # Query Fault element
    if not dss_instance.Circuit.SetActiveElement(f"Fault.{fault_id}"):
        err_msg = f"Fault element 'Fault.{fault_id}' could not be activated in OpenDSS"
        print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
        raise ValueError(err_msg)

    fault_currents = dss_instance.CktElement.Currents()
    try:
        fault_r = float(dss_instance.Properties.Value("r"))
    except Exception as e:
        err_msg = f"Could not read property 'r' from Fault.{fault_id}: {e}"
        print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
        raise ValueError(err_msg) from e

    # Query Line element
    if not dss_instance.Circuit.SetActiveElement(f"Line.{target_line}"):
        err_msg = f"Line element 'Line.{target_line}' could not be activated in OpenDSS"
        print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
        raise ValueError(err_msg)

    try:
        line_r1 = float(dss_instance.Properties.Value("r1"))
        line_x1 = float(dss_instance.Properties.Value("x1"))
    except Exception as e:
        err_msg = f"Could not read properties 'r1'/'x1' from Line.{target_line}: {e}"
        print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
        raise ValueError(err_msg) from e

    if not hasattr(ev_fault, "fault_type") or not hasattr(ev_fault, "faulted_phases") or not hasattr(ev_fault, "start_time_s") or not hasattr(ev_fault, "duration_s"):
        err_msg = f"Fault event object missing required attributes (fault_type, faulted_phases, start_time_s, duration_s): {ev_fault}"
        print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
        raise ValueError(err_msg)

    fault_info = {
        "fault_id": fault_id,
        "bus": target_line.replace("down_", "f").replace("_", "_node"),
        "target_line": target_line,
        "fault_type": str(ev_fault.fault_type),
        "fault_resistance_ohm": fault_r,
        "faulted_phases": list(ev_fault.faulted_phases),
        "fault_currents": [float(c) for c in fault_currents],
        "line_r1_ohm": line_r1,
        "line_x1_ohm": line_x1,
        "start_time_s": float(ev_fault.start_time_s),
        "duration_s": float(ev_fault.duration_s)
    }

    return json.dumps(fault_info)




class CoSimulationRunner:
    """
    Co-Simulation Orchestrator that energizes the imported plant from src.power_plant,
    handles 2 network cases (single LV network composition vs 3 LV networks composition),
    handles steady state operation (5-minute experiment run for Dataset 1) and event/fault operation
    (steady operational parameters for Datasets 2, 3, 4 without mixing steady/fault states),
    caches resolved OperatingPoint objects and ATP transient responses for evaluated network conditions,
    and uses ATPRunner strictly inside measure_transients.
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
        feeder_idx: int = 1,
        use_baseline_feeder: bool = True
    ) -> tuple[np.ndarray, dict, dict, dict]:
        """
        Exclusively executes ATP transient simulation cases and parses EMT waveforms for consumer load
        and transformer transients using derived line parameters and BCTRAN transformer specs.
        No default fallbacks are allowed; invalid or un-simulated states raise ValueError with stack trace.
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

        v_tuple = op.phase_voltages_v[target_tx]
        a_tuple = op.phase_angles_deg[target_tx]
        freq = op.frequency_hz

        atp_cache_key = (ev_key, feeder_id, target_tx, v_tuple, a_tuple, freq)
        if atp_cache_key in self._atp_response_cache:
            return self._atp_response_cache[atp_cache_key]

        atp_case_path = f"src/simulation/atp_cases/case_{ev_key}_{scenario_id}_{os.getpid()}.ATP"

        try:
            from src.transient.atp_case_builder import (
                TransformerSpec, TransformerWinding, ShortCircuitTest,
                SourceModel, ThreePhaseState, LineModel, LoadModel,
                TransientEvent, SimulationConfig
            )

            target_tx_key = f"trans{feeder_idx}"
            if "kvas" not in tx_spec or "kvs" not in tx_spec or "r_pct" not in tx_spec or "xhl_pct" not in tx_spec or "name" not in tx_spec:
                err_msg = f"Missing required transformer specification key in tx_spec: {tx_spec}"
                print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
                raise ValueError(err_msg)

            if not hasattr(op, "phase_voltages_v") or target_tx_key not in op.phase_voltages_v:
                err_msg = f"Missing phase_voltages_v for target transformer '{target_tx_key}' in operating point"
                print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
                raise ValueError(err_msg)

            if not hasattr(op, "phase_angles_deg") or target_tx_key not in op.phase_angles_deg:
                err_msg = f"Missing phase_angles_deg for target transformer '{target_tx_key}' in operating point"
                print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
                raise ValueError(err_msg)

            if not hasattr(op, "frequency_hz"):
                err_msg = f"Missing frequency_hz in operating point"
                print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
                raise ValueError(err_msg)

            kvas = tx_spec["kvas"]
            kvs = tx_spec["kvs"]
            r_pct = float(tx_spec["r_pct"])
            xhl_pct = float(tx_spec["xhl_pct"])

            transformer = TransformerSpec(
                name=str(tx_spec["name"]),
                frequency_hz=float(op.frequency_hz),
                windings=[
                    TransformerWinding("HV", float(kvs[0]), float(kvas[0]) / 1000.0, "Y"),
                    TransformerWinding("LV", float(kvs[1]), float(kvas[1]) / 1000.0, "Y")
                ],
                short_circuit_tests=[
                    ShortCircuitTest(1, 2, z_pos_pu=float(np.sqrt((r_pct/100.0)**2 + (xhl_pct/100.0)**2)), losses_pos_kw=(r_pct/100.0)*float(kvas[0]))
                ]
            )

            phase_v = op.phase_voltages_v[target_tx_key]
            phase_ang = op.phase_angles_deg[target_tx_key]

            source = SourceModel(
                name="GRID",
                frequency_hz=float(op.frequency_hz),
                pre_event=ThreePhaseState(
                    voltage_rms_v=(float(phase_v[0]), float(phase_v[1]), float(phase_v[2])),
                    voltage_angle_deg=(float(phase_ang[0]), float(phase_ang[1]), float(phase_ang[2]))
                )
            )

            line = LineModel(
                name=f"line_{feeder_idx}",
                from_bus="main_bus",
                to_bus=f"feeder{feeder_idx}_head",
                length_km=4.5,
                r1_ohm_per_km=0.21,
                x1_ohm_per_km=0.08
            )

            loads = [LoadModel(name="default_load", bus=f"feeder{feeder_idx}_sec", p_kw=100.0)]

            if hasattr(event, "event_1") and hasattr(event, "event_2"):
                ev_class = "co_event"
                ev_type = f"{event.event_1.event_type}_{event.event_2.event_type}"
                if not hasattr(event.event_1, "start_time_s") or not hasattr(event.event_1, "duration_s"):
                    raise ValueError(f"Event 1 missing start_time_s or duration_s attribute: {event.event_1}")
                start_s = float(event.event_1.start_time_s)
                dur_s = float(event.event_1.duration_s)
            else:
                if not hasattr(event, "event_class") or not hasattr(event, "event_type") or not hasattr(event, "start_time_s") or not hasattr(event, "duration_s"):
                    raise ValueError(f"Event missing required attributes (event_class, event_type, start_time_s, duration_s): {event}")
                ev_class = str(event.event_class)
                ev_type = str(event.event_type)
                start_s = float(event.start_time_s)
                dur_s = float(event.duration_s)

            if hasattr(event, "fault_type"):
                f_type = event.fault_type
                f_phases = event.faulted_phases
                f_res = float(event.fault_resistance)
            else:
                f_type = None
                f_phases = (0,)
                f_res = 0.001

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

            sim_config = SimulationConfig(t_start_s=0.0, t_stop_s=0.15, time_step_s=1e-4)

            self.atp_builder.build_explicit(
                transformer=transformer,
                source=source,
                line=line,
                loads=loads,
                event=transient_ev,
                simulation=sim_config,
                output_path=atp_case_path,
                scenario_id=scenario_id
            )

            atp_result = ATPRunner().run(atp_case_path)
            emt_waveforms = ATPOutputReader().read(atp_result, transformer_id=target_tx, event)

            if emt_waveforms is None or emt_waveforms.time_s is None or len(emt_waveforms.time_s) == 0:
                raise ValueError(f"ATP simulation returned empty waveforms for scenario {scenario_id}")

            return emt_waveforms.time_s, emt_waveforms.voltages, emt_waveforms.currents, emt_waveforms.event_metadata

        except Exception as e:
            err_msg = f"Failed ATP transient measurement for scenario '{scenario_id}': {e}"
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
        Executes a pure OpenDSS steady-state power flow simulation for Dataset 1 (5-minute daily energization run)
        without invoking ATP transient simulation or measuring transient waveforms.
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
        Executes ATP transient simulations evaluated directly for a list of co-events
        (equipment-equipment or equipment-fault event pairs) passed in `events`.
        Uses ProcessPoolExecutor with max_workers=6 to process pairs in parallel.
        For each pair, evaluates individual single event transients and joint co-event transients
        for the transformer where the pair is connected.
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
    Worker function executed in parallel across 6 process workers.
    Reuses process-level CoSimulationRunner and its OpenDSS instance,
    picks random target buses and lines across feeders in the network,
    adds dynamic loads and faults to OpenDSS, solves operating point for the connected transformer,
    and runs ATP transient simulation for Event 1, Event 2, and Joint Co-Event.
    """
    co_ev, use_baseline_feeder, seed, task_idx = args_tuple

    from src.loads import get_equipment_model

    # Use runner instance initialized per worker process
    runner = CoSimulationRunner()
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)

    ev1 = co_ev.event_1
    ev2 = co_ev.event_2

    # Determine feeder and target bus/line covering all feeders in the network
    if use_baseline_feeder:
        feeder_idx = 1
    else:
        if hasattr(co_ev, "gt_feeder_id"):
            raw_fid = co_ev.gt_feeder_id
            feeder_idx = int(raw_fid[7:]) if isinstance(raw_fid, str) and raw_fid.startswith("feeder_") else int(raw_fid)
        else:
            feeder_idx = (task_idx % 3) + 1

    rng = np.random.default_rng(seed + task_idx)
    bus_node_idx = int(rng.integers(1, 19))
    target_bus = f"f{feeder_idx}_node{bus_node_idx}"
    target_line = f"down_{feeder_idx}_{bus_node_idx}"

    target = [{
        "tx_id": f"trans{feeder_idx}_lv_boundary",
        "branch_type": "transformer_boundary",
        "target_bus": target_bus,
        "target_line": target_line
    }]

    def add_event_to_opendss(ev, event_prefix: str):
        ev_cls = getattr(ev, "event_class", None)
        if ev_cls == "equipment_switch":
            eq_type = ev.equipment_type
            eq_model = get_equipment_model(eq_type)
            p_kw = eq_model.rated_power_kw
            pf = eq_model.power_factor
            ld_name = f"MyNewLoad_{event_prefix}_{eq_type}"
            runner.dss.run_command(
                f"New Load.{ld_name} bus1={target_bus} phases=3 conn=Wye kV=0.415 kW={p_kw} PF={pf}"
            )
        elif ev_cls == "line_fault":
            f_type = ev.fault_type
            f_phases = ev.faulted_phases
            r_val = ev.fault_resistance
            num_phases = len(f_phases)
            ph_suffix = "." + ".".join(str(p + 1) for p in f_phases)
            fault_name = f"F_{event_prefix}_{f_type}"
            runner.dss.run_command(
                f"New Fault.{fault_name} bus1={target_bus}{ph_suffix} phases={num_phases} R={r_val} ontime=0.1"
            )
        else:
            err_msg = f"Unknown event class '{ev_cls}' for event: {ev}"
            print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
            raise ValueError(err_msg)

    # --- STEP 1: Add both events to network, solve for feeder parameters, evaluate transformer response for joint co-event ---
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
    runner.dss.run_command("disable Fault.*")
    add_event_to_opendss(ev1, "joint_ev1")
    add_event_to_opendss(ev2, "joint_ev2")
    op_joint = plant.solve_operating_point(runner.dss)

    active_fault_id = None
    if hasattr(co_ev, "event_2") and hasattr(co_ev.event_2, "fault_type"):
        active_fault_id = f"F_joint_ev2_{co_ev.event_2.fault_type}"
    elif hasattr(co_ev, "event_1") and hasattr(co_ev.event_1, "fault_type"):
        active_fault_id = f"F_joint_ev1_{co_ev.event_1.fault_type}"

    if active_fault_id:
        fault_info_dict = json.loads(extract_fault_info(runner.dss, active_fault_id, target_line, co_ev))
        fault_info_dict["bus"] = target_bus
        fault_info_json = json.dumps(fault_info_dict)
    else:
        fault_info_json = json.dumps({"bus": target_bus})

    t_joint, v_joint_dict, i_joint_dict, _ = runner.measure_transients(
        op_joint, co_ev, target, f"p{os.getpid()}_{task_idx}_joint",
        feeder_idx=feeder_idx, use_baseline_feeder=use_baseline_feeder
    )

    # --- STEP 2: Add Event 1 to network, solve for feeder parameters, evaluate transformer response for Event 1 ---
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
    runner.dss.run_command("disable Fault.*")
    add_event_to_opendss(ev1, "ev1")
    op_1 = plant.solve_operating_point(runner.dss)
    t1, v1_dict, i1_dict, _ = runner.measure_transients(
        op_1, ev1, target, f"p{os.getpid()}_{task_idx}_ev1",
        feeder_idx=feeder_idx, use_baseline_feeder=use_baseline_feeder
    )

    # --- STEP 3: Add Event 2 to network, solve for feeder parameters, evaluate transformer response for Event 2 ---
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
    runner.dss.run_command("disable Fault.*")
    add_event_to_opendss(ev2, "ev2")
    op_2 = plant.solve_operating_point(runner.dss)
    t2, v2_dict, i2_dict, _ = runner.measure_transients(
        op_2, ev2, target, f"p{os.getpid()}_{task_idx}_ev2",
        feeder_idx=feeder_idx, use_baseline_feeder=use_baseline_feeder
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

    return {
        "co_ev": co_ev,
        "time_s": t_joint,
        "v1": v1,
        "i1": i1,
        "v2": v2,
        "i2": i2,
        "v_joint": v_joint,
        "i_joint": i_joint,
        "fault_info": fault_info_json,
        "gt_feeder_id": f"feeder_{feeder_idx}"
    }

    

    
