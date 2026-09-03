import os
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal, Any
from pathlib import Path
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
    r_ohm: float
    l_h: float


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


def atp_e8(value: float) -> str:
    """Formats float into exact 8-character field right-aligned for ATP E8.0 format."""
    v = float(value)
    s = f"{v:8.2E}"
    if len(s) > 8:
        s = f"{v:8.1E}"
    return f"{s:>8}"


def atp_misc_card(
    dt: float,
    tmax: float,
    xopt: float = 0.0,
    copt: float = 0.0,
    epsiln: float = 0.0,
    tolmat: float = 0.0,
    tstart: float = 0.0,
) -> str:
    """Emits Card 407 (floating-point miscellaneous data card) with 7 E8.0 fields."""
    return "".join(
        atp_e8(v) for v in (dt, tmax, xopt, copt, epsiln, tolmat, tstart)
    )


def fmt_type14_source(
    node: str,
    amplitude: float,
    frequency: float,
    phase_deg: float,
    t_start: float,
    t_stop: float,
) -> str:
    """Formats Type 14 AC voltage source card following strict ATP fixed-column positions."""
    node_str = f"{node:<6}"[:6]
    return (
        f"14{node_str}"
        f"{amplitude:10.3f}"
        f"{frequency:10.3f}"
        f"{phase_deg:10.3f}"
        f"{t_start:10.3f}"
        f"{t_stop:10.3f}"
    )


def fmt_num10(v: float) -> str:
    """Formats a float value into a 10-character right-aligned field for ATP branch/switch cards."""
    v = float(v)
    if v == 0.0:
        return f"{0.0:10.4f}"
    abs_v = abs(v)
    if abs_v >= 1e5 or abs_v < 1e-3:
        s = f"{v:10.4E}"
    else:
        s = f"{v:10.4f}"
    return f"{s:>10}"[:10]


def fmt_branch(n1: str, n2: str, r: float, l_mH: float, c_uF: float) -> str:
    """Formats standard RLC branch card in ATP 80-column layout."""
    n1_s = f"{n1:<6}"[:6]
    n2_s = f"{n2:<6}"[:6] if n2 else "      "
    r_s = fmt_num10(r)
    l_s = fmt_num10(l_mH)
    c_s = fmt_num10(c_uF)
    return f"  {n1_s}{n2_s}            {r_s}{l_s}{c_s}"


def fmt_type51_branch(
    n1_a: str, n2_a: str,
    n1_b: str, n2_b: str,
    n1_c: str, n2_c: str,
    r_self: float, l_self_mH: float,
    r_mut: float, l_mut_mH: float
) -> str:
    """Formats Type 51 3-phase coupled R-L branch card in ATP 80-column layout."""
    p1a = f"{n1_a:<6}"[:6]
    p2a = f"{n2_a:<6}"[:6]
    p1b = f"{n1_b:<6}"[:6]
    p2b = f"{n2_b:<6}"[:6]
    p1c = f"{n1_c:<6}"[:6]
    p2c = f"{n2_c:<6}"[:6]
    rs_s = atp_e8(r_self)
    ls_s = atp_e8(l_self_mH)
    rm_s = atp_e8(r_mut)
    lm_s = atp_e8(l_mut_mH)
    return f"51{p1a}{p2a}{p1b}{p2b}{p1c}{p2c}      {rs_s}{ls_s}{rm_s}{lm_s}"


def fmt_switch(n1: str, n2: str, t_close: float, t_open: float) -> str:
    """Formats switch card in ATP 80-column layout."""
    n1_s = f"{n1:<6}"[:6]
    n2_s = f"{n2:<6}"[:6]
    tc_s = fmt_num10(t_close)
    to_s = fmt_num10(t_open)
    return f"  {n1_s}{n2_s}{tc_s}{to_s}"


class ATPCaseValidator:
    """Validates generated ATP card deck against column limits and card structure before execution."""

    @staticmethod
    def validate_content(atp_content: str) -> None:
        lines = atp_content.splitlines()
        errors = []
        for idx, line in enumerate(lines, 1):
            if len(line) > 80:
                errors.append(f"Line {idx} exceeds 80 characters ({len(line)} chars): {line!r}")

            if idx == 5 and not line.startswith("C") and not line.startswith("$"):
                if len(line) < 32:
                    errors.append(f"Line {idx} (misc card) is shorter than 32 characters ({len(line)} chars): {line!r}")
                for f_idx in range(min(7, len(line) // 8)):
                    field = line[f_idx * 8 : (f_idx + 1) * 8]
                    try:
                        float(field.replace("D", "E"))
                    except ValueError:
                        errors.append(f"Line {idx} field {f_idx + 1} (cols {f_idx * 8 + 1}-{f_idx * 8 + 8}) is not valid float: {field!r}")

            if line.startswith("14"):
                if len(line) < 58:
                    errors.append(f"Line {idx} type 14 source card shorter than 58 chars: {line!r}")
                node = line[2:8]
                if not node.strip():
                    errors.append(f"Line {idx} type 14 source card missing node name in cols 3-8: {line!r}")

        if errors:
            raise ValueError(f"ATP case validation failed with {len(errors)} errors:\n" + "\n".join(errors))

    @staticmethod
    def validate_file(file_path: str | Path) -> None:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"ATP case file not found for validation: {file_path}")
        content = p.read_text(encoding="utf-8", errors="ignore")
        ATPCaseValidator.validate_content(content)


class ATPCaseBuilder:
    def __init__(self, template_path: str = None):
        self.template_path = template_path

    def build_explicit(
        self,
        transformer: TransformerSpec,
        source: SourceModel,
        line: LineModel,
        loads: List[LoadModel],
        event: Any,
        simulation: SimulationConfig,
        output_path: Optional[str] = None,
        scenario_id: str = "transient_scenario"
    ) -> str:
        """
        Generates a valid ATP-EMTP card file from explicit domain models:
        (Source -> Main Bus -> Line -> Transformer -> Loads -> Fault/Switch Event).
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

        # Source peak voltages and angles from pre-event ThreePhaseState
        amp_a = source.pre_event.voltage_rms_v[0] * np.sqrt(2.0)
        amp_b = source.pre_event.voltage_rms_v[1] * np.sqrt(2.0)
        amp_c = source.pre_event.voltage_rms_v[2] * np.sqrt(2.0)

        ang_a = source.pre_event.voltage_angle_deg[0]
        ang_b = source.pre_event.voltage_angle_deg[1]
        ang_c = source.pre_event.voltage_angle_deg[2]

        src_a = fmt_type14_source("SRCA", amp_a, freq_hz, ang_a, -1.0, 1.0e3)
        src_b = fmt_type14_source("SRCB", amp_b, freq_hz, ang_b, -1.0, 1.0e3)
        src_c = fmt_type14_source("SRCC", amp_c, freq_hz, ang_c, -1.0, 1.0e3)

        branch_cards = []
        switch_cards = []

        # High resistance ground paths for source
        for ph_char in ["A", "B", "C"]:
            src_node = f"SRC{ph_char}"
            branch_cards.append(fmt_branch(src_node, "", 1e8, 0.0, 0.0))

        # Source to Main Bus internal/feeder-head connection
        for ph_char in ["A", "B", "C"]:
            src_node = f"SRC{ph_char}"
            mb_node = f"MB_{ph_char}"
            branch_cards.append(fmt_branch(src_node, mb_node, 0.001, 0.001, 0.0))

        # Line Cards (Main Bus to Transformer Primary)
        r_line = line.r1_ohm_per_km * line.length_km
        x_line = line.x1_ohm_per_km * line.length_km
        l_line_mH = (x_line / (2.0 * np.pi * freq_hz)) * 1000.0

        for ph_char in ["A", "B", "C"]:
            mb_node = f"MB_{ph_char}"
            tx_node = f"TX_{ph_char}"
            branch_cards.append(fmt_branch(mb_node, tx_node, r_line, l_line_mH, 0.0))

        # Transformer Matrix / Coupled Impedance Cards
        sc_test = transformer.short_circuit_tests[0]
        hv_w = transformer.windings[0]
        lv_w = transformer.windings[1]

        z_base = (lv_w.rated_kv ** 2 * 1000.0) / lv_w.rated_mva
        z_pos_pu = sc_test.z_pos_pu
        r_pos_pu = sc_test.losses_pos_kw / (lv_w.rated_mva * 1000.0)
        x_pos_pu = np.sqrt(max(0.0, z_pos_pu**2 - r_pos_pu**2))

        r1 = r_pos_pu * z_base
        x1 = x_pos_pu * z_base
        l1_mH = (x1 / (2.0 * np.pi * freq_hz)) * 1000.0

        if sc_test.z_zero_pu is not None and sc_test.losses_zero_kw is not None:
            z_zero_pu = sc_test.z_zero_pu
            r_zero_pu = sc_test.losses_zero_kw / (lv_w.rated_mva * 1000.0)
            x_zero_pu = np.sqrt(max(0.0, z_zero_pu**2 - r_zero_pu**2))
            r0 = r_zero_pu * z_base
            x0 = x_zero_pu * z_base
            l0_mH = (x0 / (2.0 * np.pi * freq_hz)) * 1000.0
        else:
            r0 = r1
            l0_mH = l1_mH

        r_self = (2.0 * r1 + r0) / 3.0
        l_self_mH = (2.0 * l1_mH + l0_mH) / 3.0
        r_mut = (r0 - r1) / 3.0
        l_mut_mH = (l0_mH - l1_mH) / 3.0

        # Base transformer branches
        for ph_char in ["A", "B", "C"]:
            tx_node = f"TX_{ph_char}"
            sec_node = f"SEC{ph_char}"
            branch_cards.append(fmt_branch(tx_node, sec_node, r_self, l_self_mH, 0.0))

        if abs(r_mut) > 1e-6 or abs(l_mut_mH) > 1e-6:
            type51_card = fmt_type51_branch(
                "TX_A", "SECA",
                "TX_B", "SECB",
                "TX_C", "SECC",
                r_self, l_self_mH,
                r_mut, l_mut_mH
            )
            branch_cards.append(type51_card)

        # Loads Cards (including R and L for each load)
        for l_idx, ld in enumerate(loads):
            r_val = ld.r_ohm
            l_val_mH = ld.l_h * 1000.0
            node_prefix = f"L{l_idx}"
            for ph_char in ["A", "B", "C"]:
                sec_node = f"SEC{ph_char}"
                load_node = f"{node_prefix}{ph_char}"
                branch_cards.append(fmt_branch(load_node, "", r_val, l_val_mH, 0.0))
                switch_cards.append(fmt_switch(sec_node, load_node, -1.0, 100.0))

        # Transient Event Cards (supports single events and co-events)
        events_to_card = []
        if hasattr(event, "event_1") and hasattr(event, "event_2") and event.event_1 is not None and event.event_2 is not None:
            events_to_card = [event.event_1, event.event_2]
        else:
            events_to_card = [event]

        for idx, ev in enumerate(events_to_card):
            ev_class = getattr(ev, "event_class", None)
            if ev_class is None:
                raise ValueError(f"Event object missing required attribute 'event_class': {ev}")

            start_s = float(getattr(ev, "start_time_s", 0.0))
            dur_s = float(getattr(ev, "duration_s", 0.05))
            end_s = start_s + dur_s

            if ev_class in ["line_fault", "fault"]:
                f_phases = getattr(ev, "faulted_phases", (0,))
                f_res = float(getattr(ev, "fault_resistance_ohm", getattr(ev, "fault_resistance", 0.001)))
                ph_chars = ["A", "B", "C"]
                for p_idx in f_phases:
                    ph_char = ph_chars[p_idx]
                    sec_node = f"SEC{ph_char}"
                    fault_node = f"F{idx}_{ph_char}"
                    branch_cards.append(fmt_branch(fault_node, "", f_res, 0.0, 0.0))
                    switch_cards.append(fmt_switch(sec_node, fault_node, start_s, end_s))
            elif ev_class in ["load_switch", "equipment_switch", "co_event"]:
                if not hasattr(ev, "equipment_type") or ev.equipment_type is None:
                    err_msg = f"Event missing required attribute 'equipment_type': {ev}"
                    print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
                    raise ValueError(err_msg)
                eq_type = ev.equipment_type
                from src.loads import get_equipment_model
                eq_model = get_equipment_model(eq_type)

                r_val = None
                for key in ["r_stator", "r_armature", "r_coil", "r_internal", "r_magnetron", "r_speaker"]:
                    if key in eq_model.atp_params:
                        r_val = eq_model.atp_params[key]
                        break

                l_mH_val = 0.0
                for key in ["l_armature", "l_coil", "l_ac_filter", "l_filter"]:
                    if key in eq_model.atp_params:
                        l_mH_val = float(eq_model.atp_params[key]) * 1000.0
                        break

                if l_mH_val == 0.0:
                    for key in ["x_stator", "x_rotor"]:
                        if key in eq_model.atp_params:
                            x_val = float(eq_model.atp_params[key])
                            l_mH_val = (x_val / (2.0 * np.pi * freq_hz)) * 1000.0
                            break

                c_uF_val = 0.0
                for key in ["c_doubler", "c_resonant", "c_dc_link", "c_supply_bank", "c_filter"]:
                    if key in eq_model.atp_params:
                        c_uF_val = float(eq_model.atp_params[key]) * 1e6
                        break

                if r_val is None:
                    err_msg = f"Equipment model '{eq_type}' missing required R in atp_params"
                    print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
                    raise ValueError(err_msg)

                r_eq = float(r_val)
                node_prefix = f"E{idx}"

                sw_phases = getattr(ev, "faulted_phases", (0, 1, 2))
                ph_chars = ["A", "B", "C"]
                for p_idx in sw_phases:
                    ph_char = ph_chars[p_idx]
                    sec_node = f"SEC{ph_char}"
                    load_node = f"{node_prefix}{ph_char}"
                    branch_cards.append(fmt_branch(load_node, "", r_eq, l_mH_val, c_uF_val))
                    switch_cards.append(fmt_switch(sec_node, load_node, start_s, end_s))

        misc_card = atp_misc_card(
            dt=simulation.time_step_s,
            tmax=simulation.t_stop_s,
            xopt=0.0,
            copt=0.0
        )

        atp_lines = [
            "BEGIN NEW DATA CASE",
            f"C  ATP Case File for {scenario_id}",
            f"POWER FREQUENCY                      {freq_hz:.0f}.",
            "C  dT  >< Tmax >< Xopt >< Copt ><Epsiln>",
            misc_card,
            "    1000       1       1       1       1       0       0       1       0",
            "/BRANCH",
            "C < n1 >< n2 ><ref1><ref2>< R  >< L  >< C  >",
        ] + branch_cards + [
            "BLANK BRANCH",
            "/SWITCH",
            "C < n 1>< n 2>< Tclose ><Top/Tde ><   Ie   ><Vf/CLOP ><  type  >",
        ] + switch_cards + [
            "BLANK SWITCH",
            "/SOURCE",
            "C < n 1><>< Ampl.  >< Freq.  ><Phase/T0>< TSTART >< TSTOP  >",
            src_a,
            src_b,
            src_c,
            "BLANK SOURCE",
            "/OUTPUT",
            "  SECA  SECB  SECC",
            "  TX_A  SECA  TX_B  SECB  TX_C  SECC",
            "BLANK OUTPUT",
            "BLANK PLOT",
            "BEGIN NEW DATA CASE",
            "BLANK"
        ]

        atp_content = "\n".join(atp_lines)

        ATPCaseValidator.validate_content(atp_content)

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
                r_ohm=float(ld["r_ohm"]),
                l_h=float(ld["l_h"])
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
            equipment_type=getattr(event, "equipment_type"),
            event_1=getattr(event, "event_1"),
            event_2=getattr(event, "event_2")
        )

        t_start_s = 0.0
        t_stop_s = max(start_s + dur_s + 0.05, 0.15)
        sim_config = SimulationConfig(t_start_s=t_start_s, t_stop_s=t_stop_s, time_step_s=1e-4)

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
