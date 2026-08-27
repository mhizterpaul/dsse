from opendssdirect import dss
import numpy as np
import traceback
from typing import Dict, Any, List, Optional

import src.power_plant.plant as plant
from src.transient.atp_case_builder import ATPCaseBuilder
from src.transient.atp_runner import ATPRunner
from src.transient.atp_parser import ATPOutputReader


def extract_fault_info(co_ev: Any) -> str:
    """
    Extracts JSON representation of fault parameters from a co-event.
    """
    fault_info = {}
    for ev in [getattr(co_ev, "event_1", None), getattr(co_ev, "event_2", None)]:
        if ev and getattr(ev, "event_class", "") == "line_fault":
            fault_info = {
                "fault_type": getattr(ev, "fault_type", ""),
                "fault_resistance_ohm": getattr(ev, "fault_resistance", 0.05),
                "faulted_phases": list(getattr(ev, "faulted_phases", (0,))),
                "config_id": getattr(ev, "extra_params", {}).get("config_id", "")
            }
            break
    import json
    return json.dumps(fault_info) if fault_info else ""


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
    using the ConsumerRegistry interface and OpenDSS API via single passed dss instance.
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
        if isinstance(mtr, dict):
            m_id = mtr.get("consumer_unit_id", mtr.get("consumer_id", mtr.get("boundary_unit_id")))
            bus = mtr.get("bus")
            branch_id = mtr.get("branch_id", "")
        else:
            m_id = getattr(mtr, "consumer_id", "consumer")
            bus = getattr(mtr, "bus", None)
            branch_id = getattr(mtr, "branch_id", "")

        v_mags = np.array([240.0, 240.0, 240.0])
        v_angs = np.zeros(3)
        if bus:
            dss.Circuit.SetActiveBus(bus)
            v_vec = np.array(dss.Bus.VMagAngle())
            if len(v_vec) >= 6:
                v_mags = v_vec[0::2]
                v_angs = v_vec[1::2]

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

        s_kva = float(np.sqrt(p_kw**2 + q_kvar**2))
        energy_kwh = float(p_kw * duration_hours)

        # Extract OpenDSS Feeder & Line Loss Parameters
        if "feeder1" in str(bus) or "f1" in str(bus) or "trans1" in str(m_id):
            feeder_num = "1"
        elif "feeder2" in str(bus) or "f2" in str(bus) or "trans2" in str(m_id):
            feeder_num = "2"
        elif "feeder3" in str(bus) or "f3" in str(bus) or "trans3" in str(m_id):
            feeder_num = "3"
        else:
            feeder_num = "1"

        feeder_name = f"Line.feeder{feeder_num}"
        tx_name = f"Transformer.trans{feeder_num}"

        feeder_voltage = 240.0
        feeder_current = 100.0
        feeder_line_losses = 0.0

        if dss.Circuit.SetActiveElement(feeder_name):
            f_currs = dss.CktElement.CurrentsMagAng()
            f_volts = dss.CktElement.VoltagesMagAng()
            f_losses = dss.CktElement.Losses()
            if len(f_volts) >= 2:
                feeder_voltage = round(float(np.mean(f_volts[0::2])))
            if len(f_currs) >= 2:
                feeder_current = round(float(np.mean(f_currs[0::2])))
            if len(f_losses) >= 1:
                feeder_line_losses = round(float(abs(f_losses[0]) / 1000.0))

        # Transformer losses: P_t,loss = P_core + P_cu = V^2/R_c + 3*I_t^2*R_t
        transformer_losses = 0.0
        if dss.Circuit.SetActiveElement(tx_name):
            tx_losses = dss.CktElement.Losses()
            if len(tx_losses) >= 1:
                transformer_losses = round(float(abs(tx_losses[0]) / 1000.0))

        # Consumer unit line losses: P_l,loss = 3 * I^2 * R_l = (|S|^2 / V_LL^2) * R_l
        # Query OpenDSS line branch if available, otherwise compute using 3-phase line loss formula
        line_losses = 0.0
        if branch_id and dss.Circuit.SetActiveElement(f"Line.{branch_id}"):
            c_losses = dss.CktElement.Losses()
            if len(c_losses) >= 1:
                line_losses = round(float(abs(c_losses[0]) / 1000.0), 4)

        if line_losses == 0.0:
            # 3-phase line loss calculation: P_loss = 3 * I^2 * R_line = (|S|^2 / V_LL^2) * R_line
            v_ln = float(np.mean(v_mags)) if len(v_mags) > 0 else 240.0
            v_ll = v_ln * np.sqrt(3.0)
            i_line = (s_kva * 1000.0) / (np.sqrt(3.0) * v_ll) if v_ll > 0 else 0.0
            r_line = 0.05  # ohms service line resistance
            p_loss_kw = 3.0 * (i_line ** 2) * r_line / 1000.0
            line_losses = round(float(p_loss_kw), 4)

        feeder_resistance = 0.25  # ohm/km
        feeder_inductance = round(0.35 / (2.0 * np.pi * 50.0), 6)  # H/km
        feeder_capacitance = 12.0e-9  # F/km

        meas_item = {
            "consumer_unit_id": m_id,
            "bus": bus,
            "v_mags": v_mags,
            "v_angs": v_angs,
            "p_kw": round(p_kw, 4),
            "q_kvar": round(q_kvar, 4),
            "s_kva": round(s_kva, 4),
            "energy_kwh": round(energy_kwh, 4),
            "feeder_voltage": feeder_voltage,
            "feeder_current": feeder_current,
            "feeder_resistance": feeder_resistance,
            "feeder_inductance": feeder_inductance,
            "feeder_capacitance": feeder_capacitance,
            "feeder_line_losses": feeder_line_losses,
            "transformer_losses": transformer_losses,
            "line_losses": line_losses
        }
        measurements[m_id] = meas_item

    # Measure all registered and latent consumer units directly from OpenDSS loads
    for c_unit in all_registered + all_latent:
        if c_unit.consumer_id in measurements:
            continue
        p_kw = 0.0
        q_kvar = 0.0
        for ld in c_unit.loads:
            dss.Circuit.SetActiveElement(f"load.{ld.load_id}")
            powers = dss.CktElement.Powers()
            if len(powers) >= 2:
                p_kw += sum(powers[0::2])
                q_kvar += sum(powers[1::2])

        s_kva = float(np.sqrt(p_kw**2 + q_kvar**2))
        energy_kwh = float(p_kw * duration_hours)

        v_mags = np.array([240.0, 240.0, 240.0])
        dss.Circuit.SetActiveBus(c_unit.bus_id)
        v_vec = np.array(dss.Bus.VMagAngle())
        if len(v_vec) >= 6:
            v_mags = v_vec[0::2]

        v_ln = float(np.mean(v_mags)) if len(v_mags) > 0 else 240.0
        v_ll = v_ln * np.sqrt(3.0)
        i_line = (s_kva * 1000.0) / (np.sqrt(3.0) * v_ll) if v_ll > 0 else 0.0
        r_line = 0.05
        p_loss_kw = 3.0 * (i_line ** 2) * r_line / 1000.0
        line_losses = round(float(p_loss_kw), 4)

        measurements[c_unit.consumer_id] = {
            "consumer_unit_id": c_unit.consumer_id,
            "bus": c_unit.bus_id,
            "v_mags": v_mags,
            "v_angs": np.zeros(3),
            "p_kw": round(p_kw, 4),
            "q_kvar": round(q_kvar, 4),
            "s_kva": round(s_kva, 4),
            "energy_kwh": round(energy_kwh, 4),
            "line_losses": line_losses
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
        self.dss = dss
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
        try:
            if use_single_lv_network:
                self.plant_data = plant.build_single_lv_network_composition(
                    dss=self.dss,
                    feeder_idx=1,
                    generator_p_kw=generator_p_kw,
                    generator_q_kvar=generator_q_kvar,
                    use_baseline_transformers=use_baseline_transformers,
                    loads_dict=loads,
                    seed=seed
                )
            else:
                self.plant_data = plant.build_three_lv_networks_composition(
                    dss=self.dss,
                    generator_p_kw=generator_p_kw,
                    generator_q_kvar=generator_q_kvar,
                    use_baseline_transformers=use_baseline_transformers,
                    loads_dict=loads,
                    seed=seed
                )
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
                        self.dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.{ph_num} phases=1 r={f_res}")
                    elif f_type == "LL":
                        ph1 = phases[0] + 1 
                        ph2 = phases[1] + 1 
                        self.dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.{ph1} bus2={target_bus}.{ph2} phases=1 r={f_res}")
                    else:
                        self.dss.run_command(f"new Fault.{fault_name} bus1={target_bus}.1 phases=1 r={f_res}")

        # 3. For Dataset 1 steady state run, power loads for 5 minutes (300s) to measure energy
        if is_steady_state_run or not events:
            self.dss.run_command("set stepsize=1s")
            self.dss.run_command("set number=300")
            self.dss.run_command("set mode=daily")

        op = plant.solve_operating_point(self.dss, generator_p_kw, generator_q_kvar)

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
