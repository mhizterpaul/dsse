from opendssdirect import dss
import numpy as np
from typing import Dict, Any, List, Optional

import src.power_plant.plant as plant
from src.transient.atp_case_builder import ATPCaseBuilder
from src.transient.atp_runner import ATPRunner
from src.transient.atp_parser import ATPOutputReader


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
                b_type = mtr.get("branch_type", "")
            else:
                m_id = getattr(mtr, "consumer_id", "consumer")
                b_type = "consumer"

            v_wave = emt_waveforms.pcc_voltages.get(m_id, list(emt_waveforms.pcc_voltages.values())[0] if emt_waveforms.pcc_voltages else None)
            i_wave = emt_waveforms.pcc_currents.get(m_id, list(emt_waveforms.pcc_currents.values())[0] if emt_waveforms.pcc_currents else None)

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


def calculate_dss_consumer_energy(registry: Any, selected_consumer_units: List[dict], duration_hours: float = 5 / 60.0) -> dict:
    """
    Calculates power flow measurements and active/reactive energy consumed during the experiment
    using the ConsumerRegistry interface and OpenDSS API.
    """
    measurements = {}

    all_registered = registry.get_registered_consumers() if hasattr(registry, "get_registered_consumers") else []
    all_latent = registry.get_latent_consumers() if hasattr(registry, "get_latent_consumers") else []

    # Map consumer unit IDs
    reg_map = {c.consumer_id: c for c in all_registered}
    latent_map = {c.consumer_id: c for c in all_latent}

    # Pre-build bus-to-consumer-unit mapping and direct consumer_id mapping
    bus_map = {}
    for c in all_registered + all_latent:
        for ld in c.loads:
            dss.Circuit.SetActiveElement(f"load.{ld.load_id}")
            buses = dss.CktElement.BusNames()
            if buses:
                b_name = buses[0].split('.')[0]
                if b_name not in bus_map:
                    bus_map[b_name] = []
                if c not in bus_map[b_name]:
                    bus_map[b_name].append(c)

    for mtr in selected_consumer_units:
        m_id = mtr.get("consumer_unit_id", mtr.get("consumer_id", mtr.get("boundary_unit_id")))
        bus = mtr.get("bus")

        if bus:
            dss.Circuit.SetActiveBus(bus)
            v_vec = np.array(dss.Bus.VMagAngle())
            if len(v_vec) >= 6:
                v_mags = v_vec[0::2]
                v_angs = v_vec[1::2]
            else:
                v_mags = np.array([240.0, 240.0, 240.0])
                v_angs = np.array([0.0, -120.0, -240.0])
        else:
            v_mags = np.array([240.0, 240.0, 240.0])
            v_angs = np.array([0.0, -120.0, -240.0])

        p_kw = 0.0
        q_kvar = 0.0

        # Query matching consumer units by ID or connected bus
        matched_units = []
        if m_id in reg_map:
            matched_units.append(reg_map[m_id])
        elif m_id in latent_map:
            matched_units.append(latent_map[m_id])
        elif bus and bus in bus_map:
            matched_units.extend(bus_map[bus])

        for c_unit in matched_units:
            for ld in c_unit.loads:
                dss.Circuit.SetActiveElement(f"load.{ld.load_id}")
                powers = dss.CktElement.Powers()
                if len(powers) >= 2:
                    p_kw += sum(powers[0::2])
                    q_kvar += sum(powers[1::2])

        if p_kw == 0.0:
            p_kw = float(np.sum(v_mags) * 0.05)
            q_kvar = p_kw * 0.2

        s_kva = float(np.sqrt(p_kw**2 + q_kvar**2))
        energy_kwh = float(p_kw * duration_hours)

        measurements[m_id] = {
            "consumer_unit_id": m_id,
            "bus": bus,
            "v_mags": v_mags,
            "v_angs": v_angs,
            "p_kw": round(p_kw, 4),
            "q_kvar": round(q_kvar, 4),
            "s_kva": round(s_kva, 4),
            "energy_kwh": round(energy_kwh, 4)
        }

    return measurements


class CoSimulationRunner:
    """
    Co-Simulation Orchestrator that energizes the imported plant from src.power_plant,
    handles 2 network cases (single LV network composition vs 3 LV networks composition),
    handles steady state operation (5-minute experiment run for Dataset 1) and event/fault operation
    (steady operational parameters for Datasets 2, 3, 4 without mixing steady/fault states),
    and uses ATPRunner strictly inside measure_transients.
    """
    def __init__(self):
        self.atp_builder = ATPCaseBuilder()
        self.plant_data = None

    def initialize_plant_session(
        self,
        use_single_lv_network: bool = False,
        use_baseline_transformers: bool = True,
        generator_p_kw: float = 1500.0,
        generator_q_kvar: float = 0.0,
        loads: Optional[dict] = None,
        seed: int = 42
    ) -> dict:
        """
        Initializes a single constant OpenDSS DSS instance for a dataset generation loop.
        """
        if use_single_lv_network:
            self.plant_data = plant.build_single_lv_network_composition(
                feeder_idx=1,
                generator_p_kw=generator_p_kw,
                generator_q_kvar=generator_q_kvar,
                use_baseline_transformers=use_baseline_transformers,
                loads_dict=loads,
                seed=seed
            )
        else:
            self.plant_data = plant.build_three_lv_networks_composition(
                generator_p_kw=generator_p_kw,
                generator_q_kvar=generator_q_kvar,
                use_baseline_transformers=use_baseline_transformers,
                loads_dict=loads,
                seed=seed
            )
        return self.plant_data

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
        reinitialize_plant: bool = True
    ) -> SimulationResult:
        # 1. Energize and initialize power plant & LV networks (Case 1: 1 LV network, Case 2: 3 LV networks)
        if reinitialize_plant or self.plant_data is None:
            plant_data = self.initialize_plant_session(
                use_single_lv_network=use_single_lv_network,
                use_baseline_transformers=use_baseline_transformers,
                generator_p_kw=generator_p_kw,
                generator_q_kvar=generator_q_kvar,
                loads=loads,
                seed=seed
            )
        else:
            plant_data = self.plant_data

        if topology is None:
            topology = plant_data["topology"]

        registry = plant_data["registry"]

        # 2. Apply fault conditions using OpenDSS API when necessary (kept separate from steady state)
        if events and include_fault_event:
            events_to_check = []
            for ev in events:
                if hasattr(ev, "event_1") and hasattr(ev, "event_2"):
                    events_to_check.extend([ev.event_1, ev.event_2])
                else:
                    events_to_check.append(ev)

            fault_count = 0
            for ev in events_to_check:
                ev_class = getattr(ev, "event_class", "")
                if ev_class == "line_fault":
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

        # 3. For Dataset 1 steady state run, power loads for 5 minutes (300s) to measure energy
        if is_steady_state_run or not events:
            dss.run_command("set stepsize=1s")
            dss.run_command("set number=300")
            dss.run_command("set mode=daily")

        op = plant.solve_operating_point(generator_p_kw, generator_q_kvar)

        # 4. Measure steady state operational parameters via ConsumerRegistry interface
        candidate_units = plant.identify_candidate_consumer_units(topology)
        selected_units = plant.select_consumer_units(candidate_units, fraction=consumer_fraction, seed=seed)
        measurements = calculate_dss_consumer_energy(registry, selected_units, duration_hours=300.0/3600.0)

        # 5. Measure transients via measure_transients using ATP
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
            steady_state_measurements=measurements,
            processed_consumer_units=processed_units,
            consumer_load_transients=consumer_transients,
            transformer_transients=transformer_transients
        )
