from opendssdirect import dss
import numpy as np
from typing import Dict, Any, List, Optional

from src.power_plant.plant import (
    initialize_known_plant,
    solve_operating_point,
    identify_candidate_consumer_meters,
    select_metered_consumers
)
from src.lv_networks.meters import get_consumer_measurements
from src.transient.atp_case_builder import ATPCaseBuilder
from src.transient.atp_runner import ATPRunner
from src.transient.atp_parser import ATPOutputReader


class SimulationResult:
    """
    Simulation Result container for base power flow measurements, consumer load transients,
    and transformer transients.
    """
    def __init__(
        self,
        time_s: np.ndarray,
        metered_consumers: List[dict],
        steady_state_measurements: dict,
        processed_meters: dict,
        consumer_load_transients: Optional[dict] = None,
        transformer_transients: Optional[dict] = None
    ):
        self.time_s = time_s
        self.metered_consumers = metered_consumers
        self.steady_state_measurements = steady_state_measurements
        self.processed_meters = processed_meters
        self.consumer_load_transients = consumer_load_transients or {}
        self.transformer_transients = transformer_transients or {}


class CoSimulationRunner:
    """
    Co-Simulation Orchestrator that energizes the imported plant from src.power_plant,
    powers loads for 5 minutes (300s) to make measurements using OpenDSS API as base experiment,
    handles load switching / fault conditions using OpenDSS, and encapsulates ATPRunner strictly
    inside measure_transients to record consumer load and transformer transients.
    """
    def __init__(self):
        self.atp_builder = ATPCaseBuilder()

    def measure_transients(
        self,
        op: Any,
        event: Any,
        metered_consumers: List[dict],
        scenario_id: str
    ) -> tuple[np.ndarray, dict, dict, dict]:
        """
        Dedicated measurement function in runner.py where ATPRunner is exclusively used
        to execute ATP transient simulation cases and parse EMT waveforms for consumer load
        and transformer transients.
        """
        if event is None:
            t_vec = np.linspace(0.0, 0.1, 1000)
            return t_vec, {}, {}, {}

        if hasattr(event, "event_1") and hasattr(event, "event_2"):
            t_off = getattr(event, "time_offset_s", 0.0)
            ev_key = f"{event.event_1.event_type}_{event.event_2.event_type}_coevent_{t_off:.2f}s"
        elif getattr(event, "event_class", "") == "equipment_switch":
            ev_key = f"{event.event_type}_switch"
        else:
            ev_key = "dist_fault_steady"

        atp_case_path = f"src/simulation/atp_cases/case_{ev_key}.ATP"

        class NetworkContainer:
            def __init__(self, sid):
                self.scenario_id = sid
                self.line_parameters = {"mult": 1.0}

        self.atp_builder.build(NetworkContainer(scenario_id), op, event, atp_case_path)

        # ATPRunner is exclusively invoked here inside measure_transients
        atp_result = ATPRunner().run(atp_case_path)
        emt_waveforms = ATPOutputReader().read(atp_result, metered_consumers, event)

        processed_meters = {}
        consumer_transients = {}
        transformer_transients = {}

        for mtr in metered_consumers:
            m_id = mtr.get("meter_id", mtr.get("pcc_id"))
            b_type = mtr.get("branch_type", "")
            v_wave = emt_waveforms.pcc_voltages.get(m_id, list(emt_waveforms.pcc_voltages.values())[0] if emt_waveforms.pcc_voltages else None)
            i_wave = emt_waveforms.pcc_currents.get(m_id, list(emt_waveforms.pcc_currents.values())[0] if emt_waveforms.pcc_currents else None)

            if v_wave is not None and i_wave is not None:
                data_entry = {"raw_voltage": v_wave, "raw_current": i_wave}
                processed_meters[m_id] = data_entry
                if b_type in ["transformer", "transformer_boundary"]:
                    transformer_transients[m_id] = data_entry
                else:
                    consumer_transients[m_id] = data_entry

        return emt_waveforms.time_s, processed_meters, consumer_transients, transformer_transients

    def run_simulation(
        self,
        topology: dict,
        loads: dict,
        events: Optional[List[Any]] = None,
        generator_p_kw: float = 1500.0,
        generator_q_kvar: float = 0.0,
        meter_fraction: float = 0.36,
        use_baseline_transformers: bool = True,
        include_load_event: bool = True,
        include_fault_event: bool = False,
        scenario_id: str = "scenario_0",
        seed: int = 42
    ) -> SimulationResult:
        # 1. Energize and initialize imported plant from power_plant module
        initialize_known_plant(use_baseline_transformers=use_baseline_transformers)

        dss.run_command("new linecode.down_lv nphases=3 r1=0.21 x1=0.08 r0=0.63 x0=0.24 c1=4.0 c0=2.0 units=km normamps=350.0")

        topologies = topology.get("topologies", {})
        if topologies:
            for feeder_idx, sub_topo in topologies.items():
                for ln in sub_topo.get("lines", []):
                    r1 = ln.get("r1", 0.21)
                    x1 = ln.get("x1", 0.08)
                    r0 = ln.get("r0", 0.63)
                    x0 = ln.get("x0", 0.24)
                    dss.run_command(
                        f"new line.{ln['name']} bus1={ln['bus1']} bus2={ln['bus2']} phases=3 r1={r1} x1={x1} r0={r0} x0={x0} length={ln['length']} units={ln['units']} normamps=350.0"
                    )
        else:
            for ln in topology.get("lines", []):
                r1 = ln.get("r1", 0.21)
                x1 = ln.get("x1", 0.08)
                r0 = ln.get("r0", 0.63)
                x0 = ln.get("x0", 0.24)
                dss.run_command(
                    f"new line.{ln['name']} bus1={ln['bus1']} bus2={ln['bus2']} phases=3 r1={r1} x1={x1} r0={r0} x0={x0} length={ln['length']} units={ln['units']} normamps=350.0"
                )

        # 2. Populate loads
        for ld in loads.get("loads", []):
            dss.run_command(
                f"new load.{ld['name']} bus1={ld['bus']} phases=3 kv=0.415 kw={ld['kw']} pf={ld['pf']} model={ld.get('model', 1)} status=fixed"
            )

        # 3. Apply load-switching or fault conditions using OpenDSS API
        if events:
            events_to_check = []
            for ev in events:
                if hasattr(ev, "event_1") and hasattr(ev, "event_2"):
                    events_to_check.extend([ev.event_1, ev.event_2])
                else:
                    events_to_check.append(ev)

            fault_count = 0
            for ev in events_to_check:
                ev_class = getattr(ev, "event_class", "")
                if ev_class == "line_fault" and include_fault_event:
                    fault_count += 1
                    f_type = getattr(ev, "fault_type", "LG")
                    target = getattr(ev, "target", "trans1")
                    f_res = getattr(ev, "fault_resistance", 0.05)
                    phases = getattr(ev, "faulted_phases", (0,))

                    target_bus = f"feeder{target.replace('trans', '')}_sec" if target.startswith("trans") else "feeder1_sec"
                    fault_name = f"dist_fault_{fault_count}"

                    if f_type == "LG":
                        ph_num = phases[0] + 1 if phases else 1
                        dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.{ph_num} phases=1 r={f_res}")
                    elif f_type == "LL":
                        ph1 = phases[0] + 1 if len(phases) > 0 else 1
                        ph2 = phases[1] + 1 if len(phases) > 1 else 2
                        dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.{ph1} bus2={target_bus}.{ph2} phases=1 r={f_res}")
                    else:
                        dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.1 phases=1 r={f_res}")

        # 4. Power loads for 5 minutes (300 seconds) to take measurements via DSS API
        dss.run_command("set stepsize=1s")
        dss.run_command("set number=300")
        dss.run_command("set mode=daily")

        op = solve_operating_point(generator_p_kw, generator_q_kvar)

        # 5. Measure base experiment power flow using DSS API
        candidate_meters = identify_candidate_consumer_meters(topology)
        metered_consumers = select_metered_consumers(candidate_meters, fraction=meter_fraction, seed=seed)
        measurements = get_consumer_measurements(metered_consumers)

        # 6. Exclusively measure transients via measure_transients using ATPRunner
        event = events[0] if events else None
        time_s, processed_meters, consumer_transients, transformer_transients = self.measure_transients(
            op=op,
            event=event,
            metered_consumers=metered_consumers,
            scenario_id=scenario_id
        )

        return SimulationResult(
            time_s=time_s,
            metered_consumers=metered_consumers,
            steady_state_measurements=measurements,
            processed_meters=processed_meters,
            consumer_load_transients=consumer_transients,
            transformer_transients=transformer_transients
        )
