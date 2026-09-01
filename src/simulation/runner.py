from opendssdirect import dss
import numpy as np
import traceback
import json
import os
from typing import Dict, Any, List, Optional

import src.power_plant.plant as plant
from src.transient.atp_case_builder import ATPCaseBuilder
from src.transient.atp_runner import ATPRunner
from src.transient.atp_parser import ATPOutputReader
from src.transient.events import SingleLineFaultEvent


def extract_fault_info(co_ev: Any) -> str:
    """
    Extracts JSON representation of fault parameters from a co-event using SingleLineFaultEvent
    or OpenDSS Fault element properties.
    """
    fault_info = {}

    # First check event objects (SingleLineFaultEvent) inside co_ev
    for ev in [getattr(co_ev, "event_1"), getattr(co_ev, "event_2")]:
        if isinstance(ev, SingleLineFaultEvent) or (ev and getattr(ev, "event_class") == "line_fault"):
            fault_info = {
                "fault_type": getattr(ev, "fault_type"),
                "fault_resistance_ohm": getattr(ev, "fault_resistance"),
                "faulted_phases": list(getattr(ev, "faulted_phases")),
                "target": getattr(ev, "target"),
                "config_id": getattr(ev, "parameters", {}).get("config_id")
            }
            break

    # If not found in event dataclass, try querying active OpenDSS Fault element if a circuit is active
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


class SimulationResult:
    """
    Simulation Result container for base power flow measurements, consumer load transients,
    and transformer transients evaluated using ATP and OpenDSS outputs.
    """
    def __init__(
        self,
        time_s: np.ndarray,
        selected_consumer_units: List[dict],
        steady_state_measurements: dict,
        processed_consumer_units: dict,
        consumer_load_transients: Optional[dict] = None,
        transformer_transients: Optional[dict] = None
    ):
        self.time_s = time_s
        self.selected_consumer_units = selected_consumer_units
        self.steady_state_measurements = steady_state_measurements
        self.processed_consumer_units = processed_consumer_units
        self.consumer_load_transients = consumer_load_transients or {}
        self.transformer_transients = transformer_transients or {}

    def evaluate_transients(self, emt_waveforms: Any, selected_consumer_units: List[dict]) -> tuple[dict, dict, dict]:
        """
        Evaluates consumer load transients and transformer transients using derived network parameters and ATP output.
        """
        processed_consumer_units = {}
        consumer_transients = {}
        transformer_transients = {}

        for mtr in selected_consumer_units:
            if isinstance(mtr, dict):
                m_id = mtr.get("consumer_unit_id", mtr.get("boundary_unit_id"))
                b_type = mtr.get("branch_type")
            else:
                m_id = getattr(mtr, "consumer_id")
                b_type = "consumer"

            v_wave = emt_waveforms.pcc_voltages.get(m_id, list(emt_waveforms.pcc_voltages.values())[0])
            i_wave = emt_waveforms.pcc_currents.get(m_id, list(emt_waveforms.pcc_currents.values())[0])

            if v_wave is not None and i_wave is not None:
                data_entry = {"raw_voltage": v_wave, "raw_current": i_wave}
                processed_consumer_units[m_id] = data_entry
                if b_type in ["transformer", "transformer_boundary"]:
                    transformer_transients[m_id] = data_entry
                else:
                    consumer_transients[m_id] = data_entry

        self.processed_consumer_units = processed_consumer_units
        self.consumer_load_transients = consumer_transients
        self.transformer_transients = transformer_transients
        return processed_consumer_units, consumer_transients, transformer_transients




def is_baseline_feeder_event(events: Optional[List[Any]]) -> bool:
    """
    Checks if an event targets Feeder 1 (the baseline feeder trans1 / feeder1).
    Uses explicit hasattr checks and logs warnings with stack traces when expected attributes are missing.
    """
    if not events:
        return True
    for ev in events:
        if hasattr(ev, "target"):
            target = str(ev.target)
            if "trans1" in target or "feeder1" in target or "f1" in target:
                return True
        elif hasattr(ev, "event_1") and hasattr(ev, "event_2"):
            ev1 = ev.event_1
            ev2 = ev.event_2
            t1 = str(ev1.target) if hasattr(ev1, "target") else ""
            t2 = str(ev2.target) if hasattr(ev2, "target") else ""
            if "trans1" in t1 or "feeder1" in t1 or "trans1" in t2 or "feeder1" in t2:
                return True
        else:
            print(f"WARNING: Event object '{type(ev).__name__}' missing 'target' and 'event_1' attributes.\n{''.join(traceback.format_stack())}")
    return False


class CoSimulationRunner:
    """
    Co-Simulation Orchestrator that energizes the imported plant from src.power_plant,
    handles 2 network cases (single LV network composition vs 3 LV networks composition),
    handles steady state operation (5-minute experiment run for Dataset 1) and event/fault operation
    (steady operational parameters for Datasets 2, 3, 4 without mixing steady/fault states),
    caches resolved OperatingPoint objects for evaluated network conditions,
    and uses ATPRunner strictly inside measure_transients.
    """
    def __init__(self):
        self.dss = dss
        self.atp_builder = ATPCaseBuilder()
        self.plant_data = None
        self._op_cache = {}

    def initialize_plant_session(
        self,
        use_single_lv_network: bool = False,
        use_baseline_transformers: bool = True,
        generator_p_kw: float = 1500.0,
        generator_q_kvar: float = 0.0,
        loads: Optional[dict] = None,
        seed: int = 42,
        verbose: bool = False
    ) -> dict:
        """
        Initializes a single constant OpenDSS DSS instance for a dataset generation loop.
        """
        try:
            if use_single_lv_network:
                self.plant_data = plant.build_single_lv_network_composition(
                    dss=self.dss,
                    feeder_idx=1,
                    generator_p_kw=generator_p_kw,
                    generator_q_kvar=generator_q_kvar,
                    use_baseline_transformers=use_baseline_transformers,
                    loads_dict=loads,
                    seed=seed,
                    verbose=verbose
                )
            else:
                self.plant_data = plant.build_three_lv_networks_composition(
                    dss=self.dss,
                    generator_p_kw=generator_p_kw,
                    generator_q_kvar=generator_q_kvar,
                    use_baseline_transformers=use_baseline_transformers,
                    loads_dict=loads,
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
        Appends unique scenario and PID identifiers to the case file path to guarantee process safety.
        """
        if event is None:
            t_vec = np.linspace(0.0, 0.1, 1000)
            return t_vec, {}, {}, {}

        if hasattr(event, "event_1") and hasattr(event, "event_2"):
            t_off = getattr(event, "time_offset_s")
            ev_key = f"{event.event_1.event_type}_{event.event_2.event_type}_coevent_{t_off:.2f}s"
        elif getattr(event, "event_class") == "equipment_switch":
            ev_key = f"{event.event_type}_switch"
        else:
            ev_key = "dist_fault_steady"

        # Unique ATP case file path per process and scenario to prevent race conditions
        atp_case_path = f"src/simulation/atp_cases/case_{ev_key}_{scenario_id}_{os.getpid()}.ATP"

        class NetworkContainer:
            def __init__(self, sid):
                self.scenario_id = sid
                self.line_parameters = {"mult": 1.0}

        try:
            self.atp_builder.build(NetworkContainer(scenario_id), op, event, atp_case_path)
            atp_result = ATPRunner().run(atp_case_path)
            emt_waveforms = ATPOutputReader().read(atp_result, selected_consumer_units, event)

            sim_res = SimulationResult(
                time_s=emt_waveforms.time_s,
                selected_consumer_units=selected_consumer_units,
                steady_state_measurements={},
                processed_consumer_units={}
            )

            processed_units, consumer_transients, transformer_transients = sim_res.evaluate_transients(emt_waveforms, selected_consumer_units)
            return emt_waveforms.time_s, processed_units, consumer_transients, transformer_transients
        except Exception as e:
            print(f"WARNING: Transient evaluation warning for scenario {scenario_id}: {e}\n{traceback.format_exc()}")
            t_vec = np.linspace(0.0, 0.1, 1000)
            return t_vec, {}, {}, {}

    def run_simulation(
        self,
        topology: Optional[dict] = None,
        loads: Optional[dict] = None,
        events: Optional[List[Any]] = None,
        generator_p_kw: float = 1500.0,
        generator_q_kvar: float = 0.0,
        consumer_fraction: float = 0.36,
        use_baseline_transformers: bool = True,
        use_single_lv_network: bool = False,
        include_load_event: bool = True,
        include_fault_event: bool = False,
        is_steady_state_run: bool = False,
        scenario_id: str = "scenario_0",
        seed: int = 42,
        reinitialize_plant: bool = True,
        verbose: bool = False
    ) -> SimulationResult:
        # 1. Energize and initialize power plant & LV networks (Case 1: 1 LV network, Case 2: 3 LV networks)
        if reinitialize_plant or self.plant_data is None:
            plant_data = self.initialize_plant_session(
                use_single_lv_network=use_single_lv_network,
                use_baseline_transformers=use_baseline_transformers,
                generator_p_kw=generator_p_kw,
                generator_q_kvar=generator_q_kvar,
                loads=loads,
                seed=seed,
                verbose=verbose
            )
        else:
            plant_data = self.plant_data

        if topology is None:
            topology = plant_data["topology"]

        registry = plant_data["registry"]

        # 2. Always disable existing OpenDSS Fault elements to avoid fault state leakage across sequential runs
        self.dss.run_command("disable Fault.*")

        fault_key_parts = []
        if events and include_fault_event:
            events_to_check = []
            for ev in events:
                if hasattr(ev, "event_1") and hasattr(ev, "event_2"):
                    events_to_check.extend([ev.event_1, ev.event_2])
                else:
                    events_to_check.append(ev)

            fault_count = 0
            for ev in events_to_check:
                ev_class = getattr(ev, "event_class")
                if ev_class == "line_fault":
                    fault_count += 1
                    f_type = getattr(ev, "fault_type")
                    target = getattr(ev, "target")
                    f_res = getattr(ev, "fault_resistance")
                    phases = getattr(ev, "faulted_phases")
                    fault_key_parts.append(f"{f_type}_{target}_{f_res}_{phases}")

                    target_bus = f"feeder{target.replace('trans', '')}_sec" if target.startswith("trans") else "feeder1_sec"
                    fault_name = f"dist_fault_{fault_count}"

                    ph_num = phases[0] + 1 if phases else 1
                    if f_type == "LG":
                        bus_spec = f"bus1={target_bus}.{ph_num} phases=1 r={f_res}"
                    elif f_type == "LL":
                        ph1 = phases[0] + 1 
                        ph2 = phases[1] + 1 if len(phases) > 1 else 2
                        bus_spec = f"bus1={target_bus}.{ph1} bus2={target_bus}.{ph2} phases=1 r={f_res}"
                    else:
                        bus_spec = f"bus1={target_bus}.1 phases=1 r={f_res}"

                    if self.dss.Circuit.SetActiveElement(f"Fault.{fault_name}"):
                        self.dss.run_command(f"edit Fault.{fault_name} {bus_spec} enabled=yes")
                    else:
                        self.dss.run_command(f"new Fault.{fault_name} {bus_spec}")

        # 3. For Dataset 1 steady state run, power loads for 5 minutes (300s) to measure energy
        if is_steady_state_run or not events:
            self.dss.run_command("set stepsize=1s")
            self.dss.run_command("set number=300")
            self.dss.run_command("set mode=daily")

        # Caching logic: In Dataset 4 (3 LV networks, use_baseline_transformers=False),
        # only rows targeting/using the baseline feeder (feeder 1 / trans1) reuse the baseline cached operating point.
        baseline_cache_key = ("baseline", generator_p_kw, generator_q_kvar, use_single_lv_network, tuple(fault_key_parts))
        non_baseline_cache_key = ("non_baseline", generator_p_kw, generator_q_kvar, use_single_lv_network, use_baseline_transformers, tuple(fault_key_parts))

        if use_baseline_transformers:
            if baseline_cache_key in self._op_cache:
                op = self._op_cache[baseline_cache_key]
            else:
                op = plant.solve_operating_point(self.dss, generator_p_kw, generator_q_kvar)
                self._op_cache[baseline_cache_key] = op
        else:
            # Dataset 4: Check if row targets the baseline feeder (feeder 1 / trans1)
            if is_baseline_feeder_event(events):
                if baseline_cache_key in self._op_cache:
                    op = self._op_cache[baseline_cache_key]
                else:
                    op = plant.solve_operating_point(self.dss, generator_p_kw, generator_q_kvar)
                    self._op_cache[baseline_cache_key] = op
            else:
                if non_baseline_cache_key in self._op_cache:
                    op = self._op_cache[non_baseline_cache_key]
                else:
                    op = plant.solve_operating_point(self.dss, generator_p_kw, generator_q_kvar)
                    self._op_cache[non_baseline_cache_key] = op

        # 4. Select consumer units and measure transients via measure_transients using ATP
        candidate_units = plant.identify_candidate_consumer_units(topology)
        selected_units = plant.select_consumer_units(candidate_units, fraction=consumer_fraction, seed=seed)

        event = events[0] if events else None
        time_s, processed_units, consumer_transients, transformer_transients = self.measure_transients(
            op=op,
            event=event,
            selected_consumer_units=selected_units,
            scenario_id=scenario_id
        )

        return SimulationResult(
            time_s=time_s,
            selected_consumer_units=selected_units,
            steady_state_measurements={},
            processed_consumer_units=processed_units,
            consumer_load_transients=consumer_transients,
            transformer_transients=transformer_transients
        )
