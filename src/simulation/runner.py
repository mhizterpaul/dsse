from opendssdirect import dss
import numpy as np
import traceback
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

import src.power_plant.plant as plant
from src.transient.models import (
    TransformerSpec,
    TransformerWinding,
    ShortCircuitTest,
    ThreePhaseThevenin,
    TestBranch,
    SimulationConfig,
)
from src.transient.opendss_reducer import OpenDSSReducer
from src.transient.bctran_generator import BCTRANGenerator
from src.transient.atp_case_builder import ATPCaseBuilder
from src.transient.atp_runner import ATPRunner
from src.transient.atp_parser import ATPOutputReader
from src.transient.events import (
    SingleLineFaultEvent,
    SingleEquipmentSwitchEvent,
    EquipmentEquipmentCoEvent,
    EquipmentLineFaultCoEvent,
)


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
    Returns empty JSON "{}" for non-fault co-events.
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

    # Query Fault element if present
    if dss_instance.Circuit.SetActiveElement(f"Fault.{fault_id}"):
        fault_currents = dss_instance.CktElement.Currents()
        try:
            fault_r = float(dss_instance.Properties.Value("r"))
        except Exception as e:
            fault_r = getattr(ev_fault, "fault_resistance", 0.001)
    else:
        fault_currents = [0.0] * 6
        fault_r = getattr(ev_fault, "fault_resistance", 0.001)

    # Query Line element if present
    if dss_instance.Circuit.SetActiveElement(f"Line.{target_line}"):
        try:
            line_r1 = float(dss_instance.Properties.Value("r1"))
            line_x1 = float(dss_instance.Properties.Value("x1"))
        except Exception as e:
            line_r1 = 0.01
            line_x1 = 0.05
    else:
        line_r1 = 0.01
        line_x1 = 0.05

    fault_info = {
        "fault_id": fault_id,
        "bus": target_line,
        "target_line": target_line,
        "fault_type": str(getattr(ev_fault, "fault_type", "LG")),
        "fault_resistance_ohm": fault_r,
        "faulted_phases": list(getattr(ev_fault, "faulted_phases", (0,))),
        "fault_currents": [float(c) for c in fault_currents],
        "line_r1_ohm": line_r1,
        "line_x1_ohm": line_x1,
        "start_time_s": float(getattr(ev_fault, "start_time_s", 0.02)),
        "duration_s": float(getattr(ev_fault, "duration_s", 0.5)),
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

        from src.power_plant.lv_transformers import get_distribution_transformer_spec

        tx_spec_dict = get_distribution_transformer_spec(feeder_idx, use_baseline=use_baseline_feeder)

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

        freq = op.frequency_hz
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

            # 2. Build BCTRAN Transformer Spec & generate/retrieve cached BCTRAN punch matrix
            kvas = tx_spec_dict["kvas"]
            kvs = tx_spec_dict["kvs"]
            r_pct = float(tx_spec_dict["r_pct"])
            xhl_pct = float(tx_spec_dict["xhl_pct"])
            noloadloss_pct = float(tx_spec_dict["noloadloss_pct"])
            imag_pct = float(tx_spec_dict["imag_pct"])
            conns = tx_spec_dict.get("conns", ["delta", "wye"])

            r0_pct = float(tx_spec_dict.get("r0_pct", r_pct))
            x0_pct = float(tx_spec_dict.get("x0_pct", xhl_pct))
            z0_pu = float(np.sqrt((r0_pct / 100.0) ** 2 + (x0_pct / 100.0) ** 2))
            losses_zero_kw = (r0_pct / 100.0) * float(kvas[0])

            tx_spec = TransformerSpec(
                name=str(tx_spec_dict["name"]),
                frequency_hz=float(freq),
                windings=[
                    TransformerWinding("HV", float(kvs[0]), float(kvas[0]) / 1000.0, str(conns[0]), 0.0),
                    TransformerWinding("LV", float(kvs[1]), float(kvas[1]) / 1000.0, str(conns[1]), -30.0),
                ],
                short_circuit_tests=[
                    ShortCircuitTest(
                        1,
                        2,
                        z_pos_pu=float(np.sqrt((r_pct / 100.0) ** 2 + (xhl_pct / 100.0) ** 2)),
                        losses_pos_kw=(r_pct / 100.0) * float(kvas[0]),
                        z_zero_pu=z0_pu,
                        losses_zero_kw=losses_zero_kw,
                    )
                ],
                excitation_current_percent=imag_pct,
                excitation_loss_kw=max((noloadloss_pct / 100.0) * float(kvas[0]), 25.0),
            )

            tx_cache_key = (tx_spec_dict["name"], float(freq))
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
                output_path=atp_case_path,
                scenario_id=scenario_id,
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
    Reuses process-level CoSimulationRunner and its OpenDSS instance.
    OpenDSS solves pre-event steady-state operating points of the base network.
    Test loads and faults are instantiated exclusively in ATP.
    """
    co_ev, use_baseline_feeder, seed, task_idx = args_tuple

    runner = CoSimulationRunner()
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

    # --- STEP 1: Pre-event base network operating point & joint co-event ATP simulation ---
    runner.initialize_plant_session(use_baseline_feeder=use_baseline_feeder, seed=seed)
    runner.dss.run_command("disable Fault.*")
    op_joint = plant.solve_operating_point(runner.dss)

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
    t2, v2_dict, i2_dict, _ = runner.measure_transients(
        op=op_2,
        event=ev2,
        scenario_id=f"p{os.getpid()}_{task_idx}_ev2",
        feeder_idx=feeder_idx,
        use_baseline_feeder=use_baseline_feeder,
        t_stop_override=uniform_t_stop,
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
        "gt_feeder_id": f"feeder_{feeder_idx}",
    }
