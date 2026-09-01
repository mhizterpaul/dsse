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
    processed_consumer_units: Dict[str, Any] = field(default_factory=dict)
    time_s: Optional[np.ndarray] = None
    operating_point: Optional[Any] = None


def extract_fault_info(co_ev: Any) -> str:
    """
    Extracts JSON representation of fault parameters from a co-event using SingleLineFaultEvent
    or OpenDSS Fault element properties.
    """
    fault_info = {}
     
    if not fault_info:
        try:
            if dss.Circuit.SetActiveElement("Fault.dist_fault_1"):
                r_val = float(dss.Properties.Value("r"))
                phases_val = int(dss.Properties.Value("phases"))
                bus_names = dss.CktElement.BusNames()
                fault_info = {
                    "fault_type": "LG" if phases_val == 1 else "LL",
                    "fault_resistance_ohm": r_val,
                    "faulted_phases": [0] if phases_val == 1 else [0, 1],
                    "bus": bus_names[0] 
                }
        except Exception:
            pass

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
        selected_consumer_units: List[dict],
        scenario_id: str
    ) -> tuple[np.ndarray, dict, dict, dict]:
        """
        Exclusively executes ATP transient simulation cases and parses EMT waveforms for consumer load
        and transformer transients using derived network parameters.
        Caches ATP responses for specific load or fault events under identical operating points.
        Appends unique scenario and PID identifiers to the case file path to guarantee process safety.
        """
        if event is None:
            t_vec = np.linspace(0.0, 0.1, 1000)
            return t_vec, {}, {}, {}

        # Construct deterministic feeder-aware ATP response cache key
        target_tx = getattr(event, "target", "trans1")
        feeder_id = getattr(event, "gt_feeder_id", getattr(event, "feeder_id", "feeder_1"))
        if hasattr(event, "event_1") and hasattr(event, "event_2"):
            t_off = getattr(event, "time_offset_s", 0.0)
            ev_key = f"{event.event_1.event_type}_{event.event_2.event_type}_coevent_{t_off:.4f}s"
        elif getattr(event, "event_class", "") == "equipment_switch":
            ev_key = f"{getattr(event, 'equipment_type')}_switch"
        elif getattr(event, "event_class", "") == "line_fault":
            f_phases = getattr(event, "faulted_phases", (0,))
            ev_key = f"{getattr(event, 'fault_type')}_{'-'.join(map(str, f_phases))}_{getattr(event, 'fault_resistance', 0.001)}"
        else:
            ev_key = f"event_{getattr(event, 'event_type', 'steady')}"

        v_tuple = op.phase_voltages_v.get(str(target_tx)) if hasattr(op, "phase_voltages_v") and op.phase_voltages_v else ()
        a_tuple = op.phase_angles_deg.get(str(target_tx)) if hasattr(op, "phase_angles_deg") and op.phase_angles_deg else ()
        freq = getattr(op, "frequency_hz", 50.0)

        atp_cache_key = (ev_key, feeder_id, target_tx, v_tuple, a_tuple, freq)
        if atp_cache_key in self._atp_response_cache:
            return self._atp_response_cache[atp_cache_key]

        # Unique ATP case file path per process and scenario to prevent race conditions
        atp_case_path = f"src/simulation/atp_cases/case_{ev_key}_{scenario_id}_{os.getpid()}.ATP"

        try:
            class DummyRealization:
                def __init__(self, sid):
                    self.scenario_id = sid
                    self.line_parameters = {"mult": 1.0}

            self.atp_builder.build(DummyRealization(scenario_id), op, event, atp_case_path)
            atp_result = ATPRunner().run(atp_case_path)
            emt_waveforms = ATPOutputReader().read(atp_result, selected_consumer_units, event)

            res_tuple = (emt_waveforms.time_s, emt_waveforms.pcc_voltages, emt_waveforms.pcc_currents, emt_waveforms.event_metadata)
            self._atp_response_cache[atp_cache_key] = res_tuple
            return res_tuple
        except Exception as e:
            print(f"WARNING: Transient evaluation warning for scenario {scenario_id}: {e}\n{traceback.format_exc()}")
            t_vec = np.linspace(0.0, 0.1, 1000)
            return t_vec, {}, {}, {}
            

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
        feeder_idx = getattr(co_ev, "gt_feeder_id", (task_idx % 3) + 1)
        if isinstance(feeder_idx, str) and feeder_idx.startswith("feeder_"):
            feeder_idx = int(feeder_id[7:])

    rng = np.random.default_rng(seed + task_idx)
    bus_node_idx = int(rng.integers(1, 19))
    target_bus = f"f{feeder_idx}_node{bus_node_idx}"
    target_line = f"down_{feeder_idx}_{bus_node_idx}"

    target_tx = f"trans{feeder_idx}"
    target_pcc = [{
        "consumer_unit_id": f"trans{feeder_idx}_lv_boundary_consumer_unit",
        "branch_type": "transformer_boundary",
        "target_bus": target_bus,
        "target_line": target_line
    }]

    def inject_single_event(ev, event_prefix: str):
        if getattr(ev, "event_class", "") == "equipment_switch":
            eq_type = getattr(ev, "equipment_type")
            eq_model = get_equipment_model(eq_type)
            p_kw = eq_model.rated_power_kw
            pf = eq_model.power_factor
            ld_name = f"MyNewLoad_{event_prefix}_{eq_type}"
            runner.dss.run_command(
                f"New Load.{ld_name} bus1={target_bus} phases=3 conn=Wye kV=0.415 kW={p_kw} PF={pf}"
            )
        elif getattr(ev, "event_class", "") == "line_fault":
            f_type = getattr(ev, "fault_type")
            f_phases = getattr(ev, "faulted_phases", (0,))
            r_val = getattr(ev, "fault_resistance", 0.001)
            num_phases = len(f_phases)
            ph_suffix = "." + ".".join(str(p + 1) for p in f_phases)
            fault_name = f"F_{event_prefix}_{f_type}"
            runner.dss.run_command(
                f"New Fault.{fault_name} bus1={target_bus}{ph_suffix} phases={num_phases} R={r_val} ontime=0.1"
            )

    def get_event_key(ev, prefix: str):
        if getattr(ev, "event_class", "") == "equipment_switch":
            return f"f{feeder_idx}_b{use_baseline_feeder}_eq_{prefix}_{getattr(ev, 'equipment_type')}"
        elif getattr(ev, "event_class", "") == "line_fault":
            f_phases = getattr(ev, "faulted_phases", (0,))
            return f"f{feeder_idx}_b{use_baseline_feeder}_flt_{prefix}_{getattr(ev, 'fault_type')}_{'-'.join(map(str, f_phases))}_{getattr(ev, 'fault_resistance', 0.001)}"
        return f"f{feeder_idx}_b{use_baseline_feeder}_ev_{prefix}"

    key1 = get_event_key(ev1, "ev1")
    key2 = get_event_key(ev2, "ev2")
    key_joint = f"{key1}_{key2}"

    # --- STEP 1: Joint Co-Event Simulation ---
    # First add both events to the network, solve for feeder/operating parameters under the two events,
    # and evaluate transformer response for the event duration.
    if key_joint in runner._op_cache:
        op_joint = runner._op_cache[key_joint]
    else:
        runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
        runner.dss.run_command("disable Fault.*")
        inject_single_event(ev1, "joint_ev1")
        inject_single_event(ev2, "joint_ev2")
        op_joint = plant.solve_operating_point(runner.dss)
        runner._op_cache[key_joint] = op_joint
    t_joint, v_joint_dict, i_joint_dict, _ = runner.measure_transients(op_joint, co_ev, target_pcc, f"p{os.getpid()}_{task_idx}_joint")

    # --- STEP 2: Single Event 1 Simulation ---
    # Add Event 1 to the network, solve for feeder/operating parameters under Event 1,
    # and evaluate transformer response for the event duration.
    if key1 in runner._op_cache:
        op_1 = runner._op_cache[key1]
    else:
        runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
        runner.dss.run_command("disable Fault.*")
        inject_single_event(ev1, "ev1")
        op_1 = plant.solve_operating_point(runner.dss)
        runner._op_cache[key1] = op_1
    t1, v1_dict, i1_dict, _ = runner.measure_transients(op_1, ev1, target_pcc, f"p{os.getpid()}_{task_idx}_ev1")

    # --- STEP 3: Single Event 2 Simulation ---
    # Add Event 2 to the network, solve for feeder/operating parameters under Event 2,
    # and evaluate transformer response for the event duration.
    if key2 in runner._op_cache:
        op_2 = runner._op_cache[key2]
    else:
        runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
        runner.dss.run_command("disable Fault.*")
        inject_single_event(ev2, "ev2")
        op_2 = plant.solve_operating_point(runner.dss)
        runner._op_cache[key2] = op_2
    t2, v2_dict, i2_dict, _ = runner.measure_transients(op_2, ev2, target_pcc, f"p{os.getpid()}_{task_idx}_ev2")

    tx_unit_id = f"trans{feeder_idx}_lv_boundary_consumer_unit"

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

    fault_info_json = extract_fault_info(co_ev)

    return {
        "co_ev": co_ev,
        "time_s": t_joint if t_joint is not None else np.linspace(0.0, 0.1, 1000),
        "v1": v1,
        "i1": i1,
        "v2": v2,
        "i2": i2,
        "v_joint": v_joint,
        "i_joint": i_joint,
        "fault_info": fault_info_json,
        "gt_feeder_id": f"feeder_{feeder_idx}"
    }

    

    