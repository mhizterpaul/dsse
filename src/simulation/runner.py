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
from src.transient.events import SingleLineFaultEvent, SingleEquipmentSwitchEvent, EquipmentEquipmentCoEvent


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
        use_use_baseline_feeder: bool = True,
        seed: int = 42,
        verbose: bool = False
    ) -> dict:
        """
        Initializes a single constant OpenDSS DSS instance for a dataset generation loop.
        """
        try:
            if use_use_baseline_feeder:
                self.plant_data = plant.build_single_lv_network_composition(
                    dss=self.dss,
                    seed=seed,
                    verbose=verbose
                )
            else:
                self.plant_data = plant.build_three_lv_networks_composition(
                    dss=self.dss,
                    use_use_baseline_feeder=use_use_baseline_feeder,
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

       

        try:
            self.atp_builder.build(NetworkContainer(scenario_id), op, event, atp_case_path)
            atp_result = ATPRunner().run(atp_case_path)
            emt_waveforms = ATPOutputReader().read(atp_result, selected_consumer_units, event)

            return emt_waveforms.times
        except Exception as e:
            print(f"WARNING: Transient evaluation warning for scenario {scenario_id}: {e}\n{traceback.format_exc()}")
            

    def run_steady_state_simulation(
        self,
        use_use_baseline_feeder: bool = False,
        scenario_id: str = "steady_5min_run",
        seed: int = 42,
        reinitialize_plant: bool = True,
        verbose: bool = False
    ):
        """
        Executes a pure OpenDSS steady-state power flow simulation for Dataset 1 (5-minute daily energization run)
        without invoking ATP transient simulation or measuring transient waveforms.
        """
        if reinitialize_plant or self.plant_data is None:
            plant_data = self.initialize_plant_session(
                use_use_baseline_feeder=use_use_baseline_feeder,
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

       

    def run_transient_simulation(
        self,
        events: List[Any],
        use_use_baseline_feeder: bool = True,
   
        seed: int = 42,
        reinitialize_plant: bool = True,
        verbose: bool = False
    ):
        """
        Executes an ATP transient simulation evaluated directly for a specific co-event (2 equipment events
        or an equipment and fault event pair) passed in `events`.
        """
        

        

        # 3. Measure transients via measure_transients using ATP

    

    