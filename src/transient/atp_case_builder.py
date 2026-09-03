import os
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal, Any
import numpy as np


@dataclass(frozen=True)
class TransformerWinding:
    name: str
    rated_kv: float
    rated_mva: float
    connection: str
    phase_shift_deg: float 


@dataclass(frozen=True)
class ShortCircuitTest:
    winding_i: int
    winding_j: int
    z_pos_pu: float
    losses_pos_kw: float
    z_zero_pu: Optional[float] 
    losses_zero_kw: Optional[float]


@dataclass(frozen=True)
class TransformerSpec:
    name: str
    frequency_hz: float
    windings: List[TransformerWinding]
    short_circuit_tests: List[ShortCircuitTest]
    excitation_current_percent: Optional[float] 
    excitation_loss_kw: Optional[float] 


@dataclass(frozen=True)
class ThreePhaseState:
    voltage_rms_v: Tuple[float, float, float]
    voltage_angle_deg: Tuple[float, float, float]


@dataclass(frozen=True)
class SourceModel:
    name: str
    frequency_hz: float
    pre_event: ThreePhaseState


@dataclass(frozen=True)
class LineModel:
    name: str
    from_bus: str
    to_bus: str
    length_km: float
    r1_ohm_per_km: float
    x1_ohm_per_km: float
    c1_f_per_km: float 


@dataclass(frozen=True)
class LoadModel:
    name: str
    bus: str
    p_kw: float
    q_kvar: float 
    r_ohm: Optional[float] 
    l_h: Optional[float] 


@dataclass(frozen=True)
class TransientEvent:
    event_class: str
    start_time_s: float
    duration_s: float
    location: str
    fault_type: Optional[str] 
    faulted_phases: Tuple[int, ...] 
    fault_resistance_ohm: float 
    equipment_type: Optional[str]
    event_1: Optional[Any] 
    event_2: Optional[Any] 

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s


@dataclass(frozen=True)
class SimulationConfig:
    t_start_s: float 
    t_stop_s: float 
    time_step_s: float


class ATPCaseBuilder:
    def __init__(self, template_path: str = None):
        self.template_path = template_path

    def build_explicit(
        self,
        transformer: TransformerSpec,
        source: SourceModel,
        line: LineModel,
        loads: List[LoadModel],
        event: TransientEvent,
        simulation: SimulationConfig,
        output_path: Optional[str] = None,
        scenario_id: str = "transient_scenario"
    ) -> str:
        """
        Generates a valid ATP-EMTP card file from explicit domain models:
        (Source -> Line -> BCTRAN Transformer -> Loads -> Fault/Switch Event).
        """
        if transformer is None:
            raise ValueError("TransformerSpec must be provided")
        if source is None:
            raise ValueError("SourceModel must be provided")
        if line is None:
            raise ValueError("LineModel must be provided")
        if event is None:
            raise ValueError("TransientEvent must be provided")
        if simulation is None:
            raise ValueError("SimulationConfig must be provided")

        freq_hz = transformer.frequency_hz
        freq_str = f"{freq_hz:.2f}".rjust(10)

        # Source peak voltages and angles from pre-event ThreePhaseState
        amp_a = source.pre_event.voltage_rms_v[0] * np.sqrt(2.0)
        amp_b = source.pre_event.voltage_rms_v[1] * np.sqrt(2.0)
        amp_c = source.pre_event.voltage_rms_v[2] * np.sqrt(2.0)

        ang_a = source.pre_event.voltage_angle_deg[0]
        ang_b = source.pre_event.voltage_angle_deg[1]
        ang_c = source.pre_event.voltage_angle_deg[2]

        src_a = f"14SRCA  -1{amp_a:10.2f}{freq_str}{ang_a:10.2f}          -1.0000    100.00"
        src_b = f"14SRCB  -1{amp_b:10.2f}{freq_str}{ang_b:10.2f}          -1.0000    100.00"
        src_c = f"14SRCC  -1{amp_c:10.2f}{freq_str}{ang_c:10.2f}          -1.0000    100.00"

        branch_cards = []
        switch_cards = []

        # High resistance ground paths for source
        branch_cards.extend([
            "  SRCA                      1.E8                                               0",
            "  SRCB                      1.E8                                               0",
            "  SRCC                      1.E8                                               0",
        ])

        # Line Cards
        r_line = line.r1_ohm_per_km * line.length_km
        x_line = line.x1_ohm_per_km * line.length_km
        l_line_mH = (x_line / (2.0 * np.pi * freq_hz)) * 1000.0

        r_line_str = f"{r_line:.4f}".rjust(10)
        l_line_str = f"{l_line_mH:.4f}".rjust(10)

        for ph_char in ["A", "B", "C"]:
            src_node = f"SRC{ph_char}"
            tx_node = f"TX_{ph_char}"
            branch_cards.append(f"  {src_node:<6}{tx_node:<6}             {r_line_str}{l_line_str}                                               0")

        # BCTRAN Transformer Matrix / Impedance Cards
        sc_test = transformer.short_circuit_tests[0] 
        hv_w = transformer.windings[0] 
        lv_w = transformer.windings[1] 

        z_base = (lv_w.rated_kv ** 2 * 1000.0) / lv_w.rated_mva
        z_pu = sc_test.z_pos_pu
        r_pu = sc_test.losses_pos_kw / (lv_w.rated_mva * 1000.0) 
        x_pu = np.sqrt(max(0.0, z_pu**2 - r_pu**2))

        r_tx = r_pu * z_base
        x_tx = x_pu * z_base
        l_tx_mH = (x_tx / (2.0 * np.pi * freq_hz)) * 1000.0

        r_tx_str = f"{r_tx:.4f}".rjust(10)
        l_tx_str = f"{l_tx_mH:.4f}".rjust(10)

        for ph_char in ["A", "B", "C"]:
            tx_node = f"TX_{ph_char}"
            sec_node = f"SEC{ph_char}"
            branch_cards.append(f"  {tx_node:<6}{sec_node:<6}             {r_tx_str}{l_tx_str}                                               0")

        # Loads Cards
        for l_idx, ld in enumerate(loads):
            r_val = ld.r_ohm
            if r_val is None:
                v_ll = lv_w.rated_kv * 1000.0
                p_w = ld.p_kw * 1000.0
                r_val = (v_ll ** 2) / (p_w + 1e-6)
            r_ld_str = f"{r_val:.4f}".rjust(10)
            node_prefix = f"L{l_idx}"
            for ph_char in ["A", "B", "C"]:
                sec_node = f"SEC{ph_char}"
                load_node = f"{node_prefix}{ph_char}"
                branch_cards.append(f"  {load_node:<6}                       {r_ld_str}                                               0")
                switch_cards.append(f"  {sec_node:<6}{load_node:<6}{'-1.0000':>10}{'100.00':>10}                                             0")

        # Transient Event Cards (supports single events and co-events)
        events_to_card = []
        if hasattr(event, "event_1") and hasattr(event, "event_2") and event.event_1 is not None and event.event_2 is not None:
            events_to_card = [event.event_1, event.event_2]
        else:
            events_to_card = [event]

        for idx, ev in enumerate(events_to_card):
            ev_class = getattr(ev, "event_class", getattr(event, "event_class"))
            start_s = float(getattr(ev, "start_time_s", getattr(event, "start_time_s")))
            dur_s = float(getattr(ev, "duration_s", getattr(event, "duration_s")))
            end_s = start_s + dur_s

            start_str = f"{start_s:10.4f}"
            end_str = f"{end_s:10.4f}"

            if ev_class in ["line_fault", "fault"]:
                f_phases = getattr(ev, "faulted_phases", getattr(event, "faulted_phases"))
                f_res = float(getattr(ev, "fault_resistance_ohm", getattr(ev, "fault_resistance", getattr(event, "fault_resistance_ohm"))))
                r_fault_str = f"{f_res:.4f}".rjust(10)
                ph_chars = ["A", "B", "C"]
                for p_idx in f_phases:
                    ph_char = ph_chars[p_idx]
                    sec_node = f"SEC{ph_char}"
                    fault_node = f"F{idx}_{ph_char}"
                    branch_cards.append(f"  {fault_node:<6}                       {r_fault_str}                                               0")
                    switch_cards.append(f"  {sec_node:<6}{fault_node:<6}{start_str}{end_str}                                             0")
            elif ev_class in ["load_switch", "equipment_switch", "co_event"]:
                if not hasattr(ev, "equipment_type") or ev.equipment_type is None:
                    err_msg = f"Event missing required attribute 'equipment_type': {ev}"
                    print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
                    raise ValueError(err_msg)
                eq_type = ev.equipment_type
                from src.loads import get_equipment_model
                eq_model = get_equipment_model(eq_type)
                r_stator = None
                for key in ["r_stator", "r_armature", "r_coil", "r_internal", "r_magnetron", "r_speaker"]:
                    if key in eq_model.atp_params:
                        r_stator = eq_model.atp_params[key]
                        break
                x_stator = None
                for key in ["x_stator", "l_armature", "l_coil", "l_ac_filter", "l_filter", "c_doubler", "c_resonant"]:
                    if key in eq_model.atp_params:
                        x_stator = eq_model.atp_params[key]
                        break
                if r_stator is None or x_stator is None:
                    err_msg = f"Equipment model '{eq_type}' missing required R or X in atp_params"
                    print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
                    raise ValueError(err_msg)
                r_eq = float(r_stator)
                x_eq = float(x_stator)
                r_str = f"{r_eq:.4f}".rjust(10)
                l_str = f"{x_eq * 1000.0 / (2*np.pi*freq_hz):.4f}".rjust(10)
                node_prefix = f"E{idx}"
                for ph_char in ["A", "B", "C"]:
                    sec_node = f"SEC{ph_char}"
                    load_node = f"{node_prefix}{ph_char}"
                    branch_cards.append(f"  {load_node:<6}                       {r_str}{l_str}                                               0")
                    switch_cards.append(f"  {sec_node:<6}{load_node:<6}{start_str}{end_str}                                             0")

        atp_lines = [
            "BEGIN NEW DATA CASE",
            f"C  ATP Case File for {scenario_id}",
            f"POWER FREQUENCY                      {freq_hz:.0f}.",
            "$DUMMY, XYZ000",
            "C  dT  >< Tmax >< Xopt >< Copt ><Epsiln>",
            f"{simulation.time_step_s:8.6f}{simulation.t_stop_s:8.4f}     50.     50.",
            "    1000       1       1       1       1       0       0       1       0",
            "/BRANCH",
            "C < n1 >< n2 ><ref1><ref2>< R  >< L  >< C  >",
        ] + branch_cards + [
            "/SWITCH",
            "C < n 1>< n 2>< Tclose ><Top/Tde ><   Ie   ><Vf/CLOP ><  type  >",
        ] + switch_cards + [
            "/SOURCE",
            "C < n 1><>< Ampl.  >< Freq.  ><Phase/T0><   A1   ><   T1   >< TSTART >< TSTOP  >",
            src_a,
            src_b,
            src_c,
            "/OUTPUT",
            "BLANK BRANCH",
            "BLANK SWITCH",
            "BLANK SOURCE",
            "BLANK OUTPUT",
            "/PLOT",
            "  SECA  SECB  SECC  C:TX_A-SECA C:TX_B-SECB C:TX_C-SECC",
            "BLANK PLOT",
            "BEGIN NEW DATA CASE",
            "BLANK"
        ]

        atp_content = "\n".join(atp_lines)
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(atp_content)
        return atp_content

    def build(self, realization, operating_point, event, output_path: str) -> str:
        """
        Legacy adapter interface delegating to build_explicit.
        """
        if realization is None or not hasattr(realization, "scenario_id"):
            raise ValueError("Realization must be provided with scenario_id attribute")
        scenario_id = realization.scenario_id

        if event is None:
            raise ValueError("Event must be provided to ATPCaseBuilder")

        target_tx = getattr(event, "target")
        feeder_idx = getattr(realization, "feeder_idx")

        if not operating_point:
            raise ValueError("Operating point must be provided to ATPCaseBuilder")

        freq_hz = float(getattr(operating_point, "frequency_hz"))

        tx_spec_dict = getattr(realization, "transformer_spec")
        r_pct = float(tx_spec_dict["r_pct"])
        xhl_pct = float(tx_spec_dict["xhl_pct"])
        r0_pct = float(tx_spec_dict["r0_pct"])
        x0_pct = float(tx_spec_dict["x0_pct"])
        kvas = tx_spec_dict["kvas"]
        kvs = tx_spec_dict["kvs"]
        noloadloss_pct = float(tx_spec_dict["noloadloss_pct"])
        imag_pct = float(tx_spec_dict["imag_pct"])

        z0_pu = float(np.sqrt((r0_pct/100.0)**2 + (x0_pct/100.0)**2))
        losses_zero_kw = (r0_pct / 100.0) * float(kvas[0])

        phase_v = operating_point.phase_voltages_v[target_tx]
        phase_ang = operating_point.phase_angles_deg[target_tx]

        phase_shift_hv = float(phase_ang[0])
        phase_shift_lv = float(phase_ang[0])

        transformer = TransformerSpec(
            name=str(tx_spec_dict["name"]),
            frequency_hz=freq_hz,
            windings=[
                TransformerWinding("HV", float(kvs[0]), float(kvas[0]) / 1000.0, "Y", phase_shift_hv),
                TransformerWinding("LV", float(kvs[1]), float(kvas[1]) / 1000.0, "Y", phase_shift_lv)
            ],
            short_circuit_tests=[
                ShortCircuitTest(1, 2, z_pos_pu=float(np.sqrt((r_pct/100.0)**2 + (xhl_pct/100.0)**2)), losses_pos_kw=(r_pct/100.0)*float(kvas[0]), z_zero_pu=z0_pu, losses_zero_kw=losses_zero_kw)
            ],
            excitation_current_percent=imag_pct,
            excitation_loss_kw=(noloadloss_pct / 100.0) * float(kvas[0])
        )

        source = SourceModel(
            name="GRID",
            frequency_hz=freq_hz,
            pre_event=ThreePhaseState(
                voltage_rms_v=(float(phase_v[0]), float(phase_v[1]), float(phase_v[2])),
                voltage_angle_deg=(float(phase_ang[0]), float(phase_ang[1]), float(phase_ang[2]))
            )
        )

        line_params = getattr(realization, "line_parameters")
        line = LineModel(
            name=f"line_{feeder_idx}",
            from_bus="main_bus",
            to_bus=f"feeder{feeder_idx}_head",
            length_km=float(line_params["length_km"]),
            r1_ohm_per_km=float(line_params["r1"]),
            x1_ohm_per_km=float(line_params["x1"]),
            c1_f_per_km=float(line_params["c1"])
        )

        raw_loads = getattr(realization, "loads")
        loads = [
            LoadModel(
                name=ld["name"],
                bus=ld["bus"],
                p_kw=float(ld["p_kw"]),
                q_kvar=float(ld["q_kvar"]),
                r_ohm=ld.get("r_ohm"),
                l_h=ld.get("l_h")
            )
            for ld in raw_loads
        ]

        start_s = float(getattr(event, "start_time_s"))
        dur_s = float(getattr(event, "duration_s"))
        ev_class = str(getattr(event, "event_class"))
        f_type = getattr(event, "fault_type")
        f_phases = getattr(event, "faulted_phases")
        f_res = float(getattr(event, "fault_resistance"))

        transient_ev = TransientEvent(
            event_class=ev_class,
            start_time_s=start_s,
            duration_s=dur_s,
            location=target_tx,
            fault_type=f_type,
            faulted_phases=tuple(f_phases),
            fault_resistance_ohm=f_res,
            equipment_type=getattr(event, "equipment_type", None),
            event_1=getattr(event, "event_1", None),
            event_2=getattr(event, "event_2", None)
        )

        sim_config = SimulationConfig(t_start_s, t_stop_s, time_step_s=1e-4)

        return self.build_explicit(
            transformer=transformer,
            source=source,
            line=line,
            loads=loads,
            event=transient_ev,
            simulation=sim_config,
            output_path=output_path,
            scenario_id=scenario_id
        )
