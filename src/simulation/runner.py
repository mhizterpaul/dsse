from opendssdirect import dss
import numpy as np

from src.power_plant.plant import (
    initialize_known_plant,
    solve_operating_point,
    generate_known_radial_topology,
    identify_candidate_consumer_meters,
    select_metered_consumers
)
from src.lv_networks.meters import get_consumer_measurements
from src.lv_networks.loads import distribute_loads
from src.power_plant.transformers import get_distribution_transformer_spec

from src.transient.atp_case_builder import ATPCaseBuilder
from src.transient.atp_runner import ATPRunner
from src.transient.atp_parser import ATPOutputReader

class SimulationResult:
    def __init__(self, time_s: np.ndarray, metered_consumers: list[dict], steady_state_measurements: dict, processed_meters: dict):
        self.time_s = time_s
        self.metered_consumers = metered_consumers
        self.steady_state_measurements = steady_state_measurements
        self.processed_meters = processed_meters

        # Compatibility properties
        self.metered_pccs = self.metered_consumers
        self.processed_pccs = self.processed_meters

class CoSimulationRunner:
    def __init__(self):
        self.atp_builder = ATPCaseBuilder()

    def run_scenario(self, sim_scenario, use_baseline_transformers: bool = False) -> SimulationResult:
        """
        Coordinates full co-simulation run: DSS operating point + ATP transient waveform simulation.
        Returns a structured SimulationResult object.
        """
        initialize_known_plant(use_baseline_transformers=use_baseline_transformers)

        k_net = sim_scenario.known_network
        topo = k_net.topology
        scenario_id = k_net.scenario_id

        dss.run_command(f"new linecode.down_lv nphases=3 r1=0.21 x1=0.08 r0=0.63 x0=0.24 c1=4.0 c0=2.0 units=km normamps=350.0")

        topologies = topo.get("topologies", {})
        if topologies:
            for feeder_idx, sub_topo in topologies.items():
                root_bus = sub_topo["buses"][0]
                expected_transformer_secondary = f"feeder{feeder_idx}_sec"
                assert root_bus == expected_transformer_secondary, f"LV network root {root_bus} does not match expected transformer secondary {expected_transformer_secondary}"

                for ln in sub_topo["lines"]:
                    r1 = ln.get("r1", 0.21)
                    x1 = ln.get("x1", 0.08)
                    r0 = ln.get("r0", 0.63)
                    x0 = ln.get("x0", 0.24)
                    dss.run_command(
                        f"new line.{ln['name']} bus1={ln['bus1']} bus2={ln['bus2']} phases=3 r1={r1} x1={x1} r0={r0} x0={x0} length={ln['length']} units={ln.get('units', 'km')} normamps=350.0"
                    )
        else:
            for ln in topo.get("lines", []):
                r1 = ln.get("r1", 0.21)
                x1 = ln.get("x1", 0.08)
                r0 = ln.get("r0", 0.63)
                x0 = ln.get("x0", 0.24)
                dss.run_command(
                    f"new line.{ln['name']} bus1={ln['bus1']} bus2={ln['bus2']} phases=3 r1={r1} x1={x1} r0={r0} x0={x0} length={ln['length']} units={ln.get('units', 'km')} normamps=350.0"
                )

        for ld in k_net.loads["loads"]:
            dss.run_command(
                f"new load.{ld['name']} bus1={ld['bus']} phases=3 kv=0.415 kw={ld['kw']} pf={ld['pf']} model={ld['model']} status=fixed"
            )
        for cap in k_net.loads["capacitors"]:
            dss.run_command(
                f"new capacitor.{cap['name']} bus1={cap['bus']} phases=3 kv=0.415 kvar={cap['kvar']} conn=wye"
            )
        for m in k_net.loads["motors"]:
            dss.run_command(
                f"new load.{m['name']} bus1={m['bus']} phases=3 kv=0.415 kw={m['kw']} pf={m['pf']} model=2 status=fixed"
            )
        for der in k_net.loads["ders"]:
            dss.run_command(
                f"new generator.{der['name']} bus1={der['bus']} phases=3 kv=0.415 kw={der['kw']} pf=1.0 model=1"
            )

        # Apply distribution line faults in OpenDSS prior to solving operating point
        if sim_scenario.events:
            events_to_check = []
            for ev in sim_scenario.events:
                if hasattr(ev, "event_1") and hasattr(ev, "event_2"):
                    events_to_check.extend([ev.event_1, ev.event_2])
                else:
                    events_to_check.append(ev)

            fault_count = 0
            for ev in events_to_check:
                if getattr(ev, "event_class", "") == "line_fault":
                    fault_count += 1
                    f_type = getattr(ev, "fault_type", "LG")
                    target = getattr(ev, "target", "trans1")
                    f_res = getattr(ev, "fault_resistance", 0.05)
                    phases = getattr(ev, "faulted_phases", (0,))

                    if target.startswith("trans"):
                        f_num = target.replace("trans", "")
                        target_bus = f"feeder{f_num}_sec"
                    elif not target.startswith("feeder") and not target.startswith("down_"):
                        target_bus = "feeder1_sec"
                    else:
                        target_bus = target

                    fault_name = f"dist_fault_{fault_count}"

                    if f_type == "LG":
                        ph_num = phases[0] + 1 if phases else 1
                        dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.{ph_num} phases=1 r={f_res}")
                    elif f_type == "LL":
                        ph1 = phases[0] + 1 if len(phases) > 0 else 1
                        ph2 = phases[1] + 1 if len(phases) > 1 else 2
                        dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.{ph1} bus2={target_bus}.{ph2} phases=1 r={f_res}")
                    elif f_type == "LLG":
                        dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.1.2 phases=2 r={f_res}")
                    elif f_type == "LLL":
                        dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.1.2.3 phases=3 r={f_res}")
                    else:
                        dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.1 phases=1 r={f_res}")

        op = solve_operating_point(sim_scenario.generator_p_kw, sim_scenario.generator_q_kvar)

        # 1. Identify candidate consumer meters and select metered consumers
        candidate_meters = identify_candidate_consumer_meters(topo)
        meter_fraction = getattr(sim_scenario, "meter_fraction", 0.5)
        seed = getattr(sim_scenario, "seed", 42)
        metered_consumers = select_metered_consumers(candidate_meters, fraction=meter_fraction, seed=seed)

        # 2. Get OpenDSS power flow measurements
        measurements = get_consumer_measurements(metered_consumers)

        # 3. Simulate High-Fidelity physical EMT transient waveforms using ATP adapter
        event = sim_scenario.events[0] if sim_scenario.events else None
        if event is None:
            raise RuntimeError(f"No transient event specified for scenario {scenario_id}")

        if hasattr(event, "event_1") and hasattr(event, "event_2"):
            t_off = getattr(event, "time_offset_s", 0.0)
            ev_key = f"{event.event_1.event_type}_{event.event_2.event_type}_coevent_{t_off:.2f}s"
        elif getattr(event, "event_class", "") == "equipment_switch":
            ev_key = f"{event.event_type}_switch"
        else:
            ev_key = "dist_fault_steady"

        atp_case_path = f"src/simulation/atp_cases/case_{ev_key}.ATP"
        self.atp_builder.build(k_net, op, event, atp_case_path)

        atp_result = ATPRunner().run(atp_case_path)
        emt_waveforms = ATPOutputReader().read(atp_result, metered_consumers, event)

        assert emt_waveforms is not None, f"EMT waveform generation failed for {scenario_id}"
        assert emt_waveforms.time_s.ndim == 1
        assert len(emt_waveforms.time_s) == int(10000.0 * 0.1)

        processed_meters = {}

        for mtr in metered_consumers:
            m_id = mtr.get("meter_id", mtr.get("pcc_id"))
            if mtr.get("branch_type") == "transformer" or mtr.get("branch_type") == "transformer_boundary":
                v_wave = emt_waveforms.pcc_voltages.get(m_id, list(emt_waveforms.pcc_voltages.values())[0] if emt_waveforms.pcc_voltages else None)
                i_wave = emt_waveforms.pcc_currents.get(m_id, list(emt_waveforms.pcc_currents.values())[0] if emt_waveforms.pcc_currents else None)

                if v_wave is not None and i_wave is not None:
                    processed_meters[m_id] = {
                        "raw_voltage": v_wave,
                        "raw_current": i_wave
                    }

        return SimulationResult(
            time_s=emt_waveforms.time_s,
            metered_consumers=metered_consumers,
            steady_state_measurements=measurements,
            processed_meters=processed_meters
        )
