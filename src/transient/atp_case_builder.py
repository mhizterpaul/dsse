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
    connection: str = "Y"
    phase_shift_deg: float = 0.0


@dataclass(frozen=True)
class ShortCircuitTest:
    winding_i: int
    winding_j: int
    z_pos_pu: float
    losses_pos_kw: float
    z_zero_pu: Optional[float] = None
    losses_zero_kw: Optional[float] = None


@dataclass(frozen=True)
class TransformerSpec:
    name: str
    frequency_hz: float
    windings: List[TransformerWinding]
    short_circuit_tests: List[ShortCircuitTest]
    excitation_current_percent: Optional[float] = None
    excitation_loss_kw: Optional[float] = None


@dataclass(frozen=True)
class ThreePhaseState:
    voltage_rms_v: Tuple[float, float, float]
    voltage_angle_deg: Tuple[float, float, float]


@dataclass(frozen=True)
class SourceModel:
    name: str
    frequency_hz: float
    pre_event: ThreePhaseState
    post_event: Optional[ThreePhaseState] = None


@dataclass(frozen=True)
class LineModel:
    name: str
    from_bus: str
    to_bus: str
    length_km: float
    r1_ohm_per_km: float
    x1_ohm_per_km: float
    c1_f_per_km: float = 0.0


@dataclass(frozen=True)
class LoadModel:
    name: str
    bus: str
    p_kw: float
    q_kvar: float = 0.0
    r_ohm: Optional[float] = None
    l_h: Optional[float] = None


@dataclass(frozen=True)
class TransientEvent:
    event_class: str
    start_time_s: float
    duration_s: float
    location: str
    fault_type: Optional[str] = None
    faulted_phases: Tuple[int, ...] = (0,)
    fault_resistance_ohm: float = 0.001
    equipment_type: Optional[str] = None
    event_1: Optional[Any] = None
    event_2: Optional[Any] = None

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s


@dataclass(frozen=True)
class SimulationConfig:
    t_start_s: float = 0.0
    t_stop_s: float = 0.15
    time_step_s: float = 1e-4


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
            branch_cards.append(f"  {src_node}  {tx_node}             {r_line_str}{l_line_str}                                               0")

        # BCTRAN Transformer Matrix / Impedance Cards
        sc_test = transformer.short_circuit_tests[0] if transformer.short_circuit_tests else ShortCircuitTest(1, 2, 0.0833, 18.0)
        hv_w = transformer.windings[0] if len(transformer.windings) > 0 else TransformerWinding("HV", 33.0, 1.5)
        lv_w = transformer.windings[1] if len(transformer.windings) > 1 else TransformerWinding("LV", 0.415, 1.5)

        z_base = (lv_w.rated_kv ** 2 * 1000.0) / lv_w.rated_mva
        z_pu = sc_test.z_pos_pu
        r_pu = sc_test.losses_pos_kw / (lv_w.rated_mva * 1000.0) if lv_w.rated_mva > 0 else 0.006
        x_pu = np.sqrt(max(0.0, z_pu**2 - r_pu**2))

        r_tx = r_pu * z_base
        x_tx = x_pu * z_base
        l_tx_mH = (x_tx / (2.0 * np.pi * freq_hz)) * 1000.0

        r_tx_str = f"{r_tx:.4f}".rjust(10)
        l_tx_str = f"{l_tx_mH:.4f}".rjust(10)

        for ph_char in ["A", "B", "C"]:
            tx_node = f"TX_{ph_char}"
            sec_node = f"SEC{ph_char}"
            branch_cards.append(f"  {tx_node}  {sec_node}             {r_tx_str}{l_tx_str}                                               0")

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
            ev_class = getattr(ev, "event_class", getattr(event, "event_class", "equipment_switch"))
            start_s = float(getattr(ev, "start_time_s", getattr(event, "start_time_s", 0.02)))
            dur_s = float(getattr(ev, "duration_s", getattr(event, "duration_s", 0.5)))
            end_s = start_s + dur_s

            start_str = f"{start_s:10.4f}"
            end_str = f"{end_s:10.4f}"

            if ev_class in ["line_fault", "fault"]:
                f_phases = getattr(ev, "faulted_phases", getattr(event, "faulted_phases", (0,)))
                f_res = float(getattr(ev, "fault_resistance_ohm", getattr(ev, "fault_resistance", getattr(event, "fault_resistance_ohm", 0.001))))
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
            "  SECA  SECB  SECC",
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

        target_tx = getattr(event, "target", "trans1")
        feeder_idx = getattr(realization, "feeder_idx", 1)

        if not operating_point:
            raise ValueError("Operating point must be provided to ATPCaseBuilder")

        freq_hz = float(getattr(operating_point, "frequency_hz", 50.0))

        tx_spec_dict = getattr(realization, "transformer_spec", {})
        r_pct = float(tx_spec_dict.get("r_pct", 0.60))
        xhl_pct = float(tx_spec_dict.get("xhl_pct", 8.33))
        kvas = tx_spec_dict.get("kvas", [1500.0, 1500.0])
        kvs = tx_spec_dict.get("kvs", [11.0, 0.415])

        transformer = TransformerSpec(
            name=str(tx_spec_dict.get("name", f"trans{feeder_idx}")),
            frequency_hz=freq_hz,
            windings=[
                TransformerWinding("HV", kvs[0], kvas[0] / 1000.0, "Y"),
                TransformerWinding("LV", kvs[1], kvas[1] / 1000.0, "Y")
            ],
            short_circuit_tests=[
                ShortCircuitTest(1, 2, z_pos_pu=np.sqrt((r_pct/100.0)**2 + (xhl_pct/100.0)**2), losses_pos_kw=(r_pct/100.0)*kvas[0])
            ]
        )

        phase_v = operating_point.phase_voltages_v.get(target_tx, (240.0, 240.0, 240.0)) if hasattr(operating_point, "phase_voltages_v") else (240.0, 240.0, 240.0)
        phase_ang = operating_point.phase_angles_deg.get(target_tx, (0.0, -120.0, 120.0)) if hasattr(operating_point, "phase_angles_deg") else (0.0, -120.0, 120.0)

        source = SourceModel(
            name="GRID",
            frequency_hz=freq_hz,
            pre_event=ThreePhaseState(
                voltage_rms_v=(float(phase_v[0]), float(phase_v[1]), float(phase_v[2])),
                voltage_angle_deg=(float(phase_ang[0]), float(phase_ang[1]), float(phase_ang[2]))
            )
        )

        line_params = getattr(realization, "line_parameters", {})
        line = LineModel(
            name=f"line_{feeder_idx}",
            from_bus="main_bus",
            to_bus=f"feeder{feeder_idx}_head",
            length_km=4.5,
            r1_ohm_per_km=float(line_params.get("r1", 0.21)),
            x1_ohm_per_km=float(line_params.get("x1", 0.08))
        )

        loads = [LoadModel(name="default_load", bus=f"feeder{feeder_idx}_sec", p_kw=100.0)]

        start_s = float(getattr(event, "start_time_s", 0.02))
        dur_s = float(getattr(event, "duration_s", 0.5))
        ev_class = str(getattr(event, "event_class", "equipment_switch"))
        f_type = getattr(event, "fault_type", None)
        f_phases = getattr(event, "faulted_phases", (0,))
        f_res = float(getattr(event, "fault_resistance", 0.001))

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

        sim_config = SimulationConfig(t_start_s=0.0, t_stop_s=0.15, time_step_s=1e-4)

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
