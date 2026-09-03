import os
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal, Any
from pathlib import Path
import numpy as np


def opendss_kv_to_bctran_winding_kv(rated_ll_kv: float, connection: str) -> float:
    """Converts 3-phase line-to-line kV rating to winding rated voltage based on connection type."""
    conn = connection.strip().upper()
    if conn in ["Y", "WYE", "LN"]:
        return rated_ll_kv / np.sqrt(3.0)
    if conn in ["D", "DELTA", "LL"]:
        return rated_ll_kv
    raise ValueError(f"Unsupported transformer connection: {connection}")


Connection = Literal["Y", "D", "WYE", "DELTA"]


@dataclass(frozen=True)
class TransformerWinding:
    name: str
    rated_ll_kv: float
    rated_mva: float
    connection: Connection
    tap_pu: float = 1.0


@dataclass(frozen=True)
class ShortCircuitTest:
    winding_i: int
    winding_j: int
    z_pos_pu: float
    losses_pos_kw: float
    test_mva: float
    z_zero_pu: Optional[float] = None
    losses_zero_kw: Optional[float] = None
    zero_test_mva: Optional[float] = None


@dataclass(frozen=True)
class TransformerSpec:
    name: str
    frequency_hz: float
    windings: Tuple[TransformerWinding, ...]
    short_circuit_tests: Tuple[ShortCircuitTest, ...]
    excitation_current_percent: Optional[float] = None
    excitation_loss_kw: Optional[float] = None
    vector_group: Optional[str] = "Dy11"


@dataclass(frozen=True)
class ThreePhaseState:
    voltage_rms_v: Tuple[float, float, float]
    voltage_angle_deg: Tuple[float, float, float]


@dataclass(frozen=True)
class SourceImpedance:
    r1_ohm: float
    x1_ohm: float
    r0_ohm: Optional[float] = None
    x0_ohm: Optional[float] = None


@dataclass(frozen=True)
class TheveninEquivalent:
    boundary_bus: str
    frequency_hz: float
    v_th_rms_v: Tuple[float, float, float]
    v_th_angle_deg: Tuple[float, float, float]
    z_th_ohm: np.ndarray  # (3, 3) complex phase-domain matrix
    z1_ohm: complex
    z0_ohm: Optional[complex] = None
    z2_ohm: Optional[complex] = None


@dataclass(frozen=True)
class NetworkEquivalent:
    boundary_bus: str
    frequency_hz: float
    vabc_rms_v: Tuple[float, float, float]
    vabc_angle_deg: Tuple[float, float, float]
    z1_ohm: complex
    z0_ohm: Optional[complex] = None
    z2_ohm: Optional[complex] = None


@dataclass(frozen=True)
class SourceModel:
    name: str
    frequency_hz: float
    pre_event: ThreePhaseState
    impedance: Optional[SourceImpedance] = None
    network_equivalent: Optional[NetworkEquivalent] = None


@dataclass(frozen=True)
class LineModel:
    name: str
    from_bus: str
    to_bus: str
    length_km: float
    r1_ohm_per_km: float
    x1_ohm_per_km: float
    c1_f_per_km: float
    r0_ohm_per_km: float = 0.0
    x0_ohm_per_km: float = 0.0
    c0_f_per_km: float = 0.0


@dataclass(frozen=True)
class ReducedLoad:
    load_id: str
    name: str
    bus: str
    phases: Tuple[int, ...]
    r_ohm_phase: Tuple[float, float, float]
    l_h_phase: Tuple[float, float, float]
    pre_event_connected: bool = True


@dataclass(frozen=True)
class LoadModel:
    name: str
    bus: str
    p_kw: float
    q_kvar: float
    r_ohm: float
    l_h: float


@dataclass(frozen=True)
class LoadSwitchEvent:
    event_id: str
    start_time_s: float
    duration_s: float
    load_ids: Tuple[str, ...]
    action: Literal["open", "close"] = "close"
    equipment_type: Optional[str] = None


@dataclass(frozen=True)
class LineFaultEvent:
    event_id: str
    start_time_s: float
    duration_s: float
    location_fraction: float
    fault_type: Literal["LG", "LL", "LLG", "LLL"]
    faulted_phases: Tuple[int, ...]
    fault_resistance_ohm: float


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
    location_fraction: float = 0.5

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


def atp_e16(value: float) -> str:
    """Formats float into exact 16-character field right-aligned for ATP $VINTAGE, 1 E16.8 format."""
    s = f"{float(value):16.8E}"
    if len(s) > 16:
        s = f"{float(value):16.7E}"
    return f"{s:>16}"[:16]


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
    """Formats Type 14 AC voltage source card following strict ATP fixed-column positions (80 cols)."""
    node_str = f"{node:<6}"[:6]
    amp_s = f"{amplitude:10.3f}"
    freq_s = f"{frequency:10.3f}"
    ph_s = f"{phase_deg:10.3f}"
    a1_s = " " * 10
    t1_s = " " * 10
    tstart_s = f"{t_start:10.3f}"
    tstop_s = f"{t_stop:10.3f}"
    return f"14{node_str} 0{amp_s}{freq_s}{ph_s}{a1_s}{t1_s}{tstart_s}{tstop_s}".ljust(80)


def fmt_branch(
    n1: str,
    n2: str,
    r: float,
    l_mH: float,
    c_uF: float,
) -> str:
    """
    ATP uncoupled type-0 series RLC branch in standard ATP fixed-column format (80 cols).
    """
    if not any(abs(float(v)) > 0.0 for v in (r, l_mH, c_uF)):
        raise ValueError(f"ATP branch {n1}->{n2} has R=L=C=0; ATP requires at least one non-zero parameter.")

    bus1 = f"{n1:<6}"[:6]
    bus2 = f"{n2:<6}"[:6] if n2 else " " * 6
    bus3 = " " * 6
    bus4 = " " * 6

    r_s = f"{r:10.6f}" if abs(r) < 1e4 else f"{r:10.2E}"
    l_s = f"{l_mH:10.6f}" if abs(l_mH) < 1e4 else f"{l_mH:10.2E}"
    c_s = f"{c_uF:10.4f}" if abs(c_uF) < 1e4 else f"{c_uF:10.2E}"

    return (
        "  "
        f"{bus1}"
        f"{bus2}"
        f"{bus3}"
        f"{bus4}"
        f"{r_s:>10}"
        f"{l_s:>10}"
        f"{c_s:>10}"
    ).ljust(80)


def fmt_switch(
    n1: str,
    n2: str,
    t_close: float,
    t_open: float,
) -> str:
    """Formats switch card in ATP fixed-column layout starting in column 3 (80 cols)."""
    n1_s = f"{n1:<6}"[:6]
    n2_s = f"{n2:<6}"[:6]

    return (
        "  "
        f"{n1_s}"
        f"{n2_s}"
        f"{t_close:10.4f}"
        f"{t_open:10.4f}"
        f"{0.0:10.4f}"
        f"{0.0:10.4f}"
        f"{0:10.0f}"
    ).ljust(80)


class ATPCaseValidator:
    """Validates generated ATP card deck against column limits and card structure before execution."""

    @staticmethod
    def validate_content(atp_content: str) -> None:
        lines = atp_content.splitlines()
        errors = []
        for idx, line in enumerate(lines, 1):
            if len(line) > 80:
                errors.append(f"Line {idx} exceeds 80 characters ({len(line)} chars): {line!r}")

            if line.startswith("14"):
                if len(line) < 70:
                    errors.append(f"Line {idx} type 14 source card shorter than 70 chars: {line!r}")
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
        loads: Any,
        event: Any,
        simulation: SimulationConfig,
        output_path: Optional[str] = None,
        scenario_id: str = "transient_scenario"
    ) -> str:
        """
        Generates a valid ATP-EMTP card file from explicit domain models:
        (Source -> Main Bus -> Line -> Transformer -> Pre-event Loads -> Event Overlay).
        Node names strictly adhere to <=6 character ATP limit (SRCA, MBA, TXA, SCA, etc).
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

        v_rms_a = source.pre_event.voltage_rms_v[0]
        v_rms_b = source.pre_event.voltage_rms_v[1]
        v_rms_c = source.pre_event.voltage_rms_v[2]

        amp_a = v_rms_a * np.sqrt(2.0)
        amp_b = v_rms_b * np.sqrt(2.0)
        amp_c = v_rms_c * np.sqrt(2.0)

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

        # Source to Main Bus connection
        r_src = source.impedance.r1_ohm if source.impedance else 0.001
        x_src = source.impedance.x1_ohm if source.impedance else 0.001
        l_src_mH = (x_src / (2.0 * np.pi * freq_hz)) * 1000.0
        for ph_char in ["A", "B", "C"]:
            src_node = f"SRC{ph_char}"
            mb_node = f"MB{ph_char}"
            branch_cards.append(fmt_branch(src_node, mb_node, r_src, l_src_mH, 0.0))

        # Line Cards (Main Bus to Transformer Primary TXA, TXB, TXC)
        r_line_tot = line.r1_ohm_per_km * line.length_km
        x_line_tot = line.x1_ohm_per_km * line.length_km

        events_to_card = []
        if hasattr(event, "event_1") and hasattr(event, "event_2") and event.event_1 is not None and event.event_2 is not None:
            events_to_card = [event.event_1, event.event_2]
        else:
            events_to_card = [event]

        has_line_fault = any(getattr(ev, "event_class", None) in ["line_fault", "fault"] or isinstance(ev, LineFaultEvent) for ev in events_to_card)
        alpha = float(getattr(event, "location_fraction", 0.5)) if has_line_fault else 1.0
        alpha = min(max(alpha, 0.05), 0.95) if has_line_fault else 1.0

        if has_line_fault and alpha < 1.0:
            r1_m = max(1e-4, r_line_tot * alpha)
            l1_mH = max(1e-4, ((x_line_tot * alpha) / (2.0 * np.pi * freq_hz)) * 1000.0)
            r2_m = max(1e-4, r_line_tot * (1.0 - alpha))
            l2_mH = max(1e-4, ((x_line_tot * (1.0 - alpha)) / (2.0 * np.pi * freq_hz)) * 1000.0)

            for ph_char in ["A", "B", "C"]:
                mb_node = f"MB{ph_char}"
                floc_node = f"FL{ph_char}"
                tx_node = f"TX{ph_char}"
                branch_cards.append(fmt_branch(mb_node, floc_node, r1_m, l1_mH, 0.0))
                branch_cards.append(fmt_branch(floc_node, tx_node, r2_m, l2_mH, 0.0))
        else:
            l_line_mH = (x_line_tot / (2.0 * np.pi * freq_hz)) * 1000.0
            for ph_char in ["A", "B", "C"]:
                mb_node = f"MB{ph_char}"
                tx_node = f"TX{ph_char}"
                branch_cards.append(fmt_branch(mb_node, tx_node, r_line_tot, l_line_mH, 0.0))

        # Transformer Primary (TX) to Secondary (SC) using Sequence Coupled Type 51, 52, 53 representation
        # On LV secondary 0.415 kV LL base (0.2396 kV LN base)
        sc_test = transformer.short_circuit_tests[0]
        hv_w = transformer.windings[0]
        lv_w = transformer.windings[1]

        lv_ll_kv = getattr(lv_w, "rated_ll_kv", getattr(lv_w, "rated_kv", 0.415))
        z_base = (lv_ll_kv ** 2 * 1000.0) / lv_w.rated_mva
        z_pos_pu = sc_test.z_pos_pu
        r_pos_pu = sc_test.losses_pos_kw / (lv_w.rated_mva * 1000.0)
        x_pos_pu = np.sqrt(max(0.0, z_pos_pu**2 - r_pos_pu**2))

        r_pos = r_pos_pu * z_base
        x_pos = x_pos_pu * z_base
        l_pos_mH = (x_pos / (2.0 * np.pi * freq_hz)) * 1000.0

        if sc_test.z_zero_pu is not None and sc_test.losses_zero_kw is not None:
            z_zero_pu = sc_test.z_zero_pu
            r_zero_pu = sc_test.losses_zero_kw / (lv_w.rated_mva * 1000.0)
            x_zero_pu = np.sqrt(max(0.0, z_zero_pu**2 - r_zero_pu**2))
            r_zero = r_zero_pu * z_base
            x_zero = x_zero_pu * z_base
            l_zero_mH = (x_zero / (2.0 * np.pi * freq_hz)) * 1000.0
        else:
            r_zero = r_pos * 2.0
            l_zero_mH = l_pos_mH * 0.5

        # Type 51 card defines Z0 (R0, L0, C0) and Type 52 card defines Z1 (R1, L1, C1)
        c1 = f"51TXA   SCA                 {r_zero:10.6f}{l_zero_mH:10.6f}{0.0:10.4f}".ljust(80)
        c2 = f"52TXB   SCB                 {r_pos:10.6f}{l_pos_mH:10.6f}{0.0:10.4f}".ljust(80)
        c3 = "53TXC   SCC".ljust(80)
        branch_cards.extend([c1, c2, c3])

        # --- PRE-EVENT REDUCED LOADS & LOAD SWITCHING OVERLAY ---
        v_phase_list = [v_rms_a, v_rms_b, v_rms_c]

        # Check if load switch events are present
        load_sw_events = [ev for ev in events_to_card if getattr(ev, "event_class", None) in ["load_switch", "equipment_switch", "co_event"] or isinstance(ev, LoadSwitchEvent)]

        for l_idx, ld in enumerate(loads):
            node_prefix = f"L{l_idx:02d}"
            for p_idx, ph_char in enumerate(["A", "B", "C"]):
                v_ph = v_phase_list[p_idx] if v_phase_list[p_idx] > 0 else 239.6
                if isinstance(ld, ReducedLoad):
                    r_val = ld.r_ohm_phase[p_idx]
                    l_val_mH = ld.l_h_phase[p_idx] * 1000.0
                else:
                    if getattr(ld, "p_kw", 0.0) > 0.0 or getattr(ld, "q_kvar", 0.0) > 0.0:
                        p_phase_w = max(1e-3, (ld.p_kw * 1000.0) / 3.0)
                        q_phase_var = (ld.q_kvar * 1000.0) / 3.0
                        s2_phase = p_phase_w**2 + q_phase_var**2
                        r_val = (v_ph**2 * p_phase_w) / s2_phase
                        x_val = (v_ph**2 * q_phase_var) / s2_phase
                        l_val_mH = max(0.0, (x_val / (2.0 * np.pi * freq_hz)) * 1000.0)
                    elif getattr(ld, "r_ohm", 0.0) > 0.0:
                        r_val = ld.r_ohm
                        l_val_mH = ld.l_h * 1000.0
                    else:
                        r_val = 10.0
                        l_val_mH = 1.0

                sec_node = f"SC{ph_char}"
                load_node = f"{node_prefix}{ph_char}"
                branch_cards.append(fmt_branch(load_node, "", r_val, l_val_mH, 0.0))

                # Load branches are connected from t_close = -1.0 s for pre-event steady-state initialization
                sw_t_close = -1.0
                sw_t_open = 100.0
                for sw_ev in load_sw_events:
                    sw_start = float(getattr(sw_ev, "start_time_s", 0.05))
                    sw_dur = float(getattr(sw_ev, "duration_s", 0.05))
                    sw_action = str(getattr(sw_ev, "action", "close"))
                    target_ids = getattr(sw_ev, "load_ids", tuple())

                    if (target_ids and (str(l_idx) in target_ids or ld.name in target_ids)) or (not target_ids and l_idx == len(loads) - 1):
                        if sw_action == "open":
                            sw_t_close = -1.0
                            sw_t_open = sw_start
                        else:
                            sw_t_close = sw_start
                            sw_t_open = sw_start + sw_dur

                switch_cards.append(fmt_switch(sec_node, load_node, sw_t_close, sw_t_open))

        # --- EXPLICIT TEST EQUIPMENT LOAD OVERLAY ---
        for idx, sw_ev in enumerate(load_sw_events):
            eq_type = getattr(sw_ev, "equipment_type", None)
            if eq_type:
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

                if r_val is not None:
                    sw_start = float(getattr(sw_ev, "start_time_s", 0.05))
                    sw_dur = float(getattr(sw_ev, "duration_s", 0.05))
                    sw_action = str(getattr(sw_ev, "action", "close"))

                    if sw_action == "close":
                        sw_t_close = sw_start
                        sw_t_open = sw_start + sw_dur
                    else:
                        sw_t_close = -1.0
                        sw_t_open = sw_start

                    node_prefix = f"E{idx:02d}"
                    sw_phases = getattr(sw_ev, "faulted_phases", (0, 1, 2))
                    ph_chars = ["A", "B", "C"]
                    for p_idx in sw_phases:
                        ph_char = ph_chars[p_idx]
                        sec_node = f"SC{ph_char}"
                        load_node = f"{node_prefix}{ph_char}"
                        branch_cards.append(fmt_branch(load_node, "", float(r_val), l_mH_val, c_uF_val))
                        switch_cards.append(fmt_switch(sec_node, load_node, sw_t_close, sw_t_open))

        # --- LINE FAULT OVERLAY ---
        fault_events = [ev for ev in events_to_card if getattr(ev, "event_class", None) in ["line_fault", "fault"] or isinstance(ev, LineFaultEvent)]
        for idx, ev in enumerate(fault_events):
            start_s = float(getattr(ev, "start_time_s", 0.0))
            dur_s = float(getattr(ev, "duration_s", 0.05))
            end_s = start_s + dur_s

            f_type = str(getattr(ev, "fault_type", "LG")).upper()
            f_phases = getattr(ev, "faulted_phases", (0,))
            f_res = float(getattr(ev, "fault_resistance_ohm", getattr(ev, "fault_resistance", 0.001)))
            ph_chars = ["A", "B", "C"]

            node_base = "FL" if has_line_fault and alpha < 1.0 else "SC"

            if f_type in ["LG", "AG", "BG", "CG"]:
                for p_idx in f_phases:
                    ph_char = ph_chars[p_idx]
                    bus_node = f"{node_base}{ph_char}"
                    fault_node = f"F{idx}{ph_char}"
                    branch_cards.append(fmt_branch(fault_node, "", f_res, 0.0, 0.0))
                    switch_cards.append(fmt_switch(bus_node, fault_node, start_s, end_s))
            elif f_type in ["LL", "AB", "BC", "CA"]:
                if len(f_phases) >= 2:
                    p1_char = ph_chars[f_phases[0]]
                    p2_char = ph_chars[f_phases[1]]
                    f_node1 = f"F{idx}1"
                    f_node2 = f"F{idx}2"
                    branch_cards.append(fmt_branch(f_node1, f_node2, f_res, 0.0, 0.0))
                    switch_cards.append(fmt_switch(f"{node_base}{p1_char}", f_node1, start_s, end_s))
                    switch_cards.append(fmt_switch(f"{node_base}{p2_char}", f_node2, start_s, end_s))
            else:
                for p_idx in f_phases:
                    ph_char = ph_chars[p_idx]
                    bus_node = f"{node_base}{ph_char}"
                    fault_node = f"F{idx}{ph_char}"
                    branch_cards.append(fmt_branch(fault_node, "", f_res, 0.0, 0.0))
                    switch_cards.append(fmt_switch(bus_node, fault_node, start_s, end_s))

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
            "C < n1 >< n2 ><ref1><ref2>< R       >< L       >< C       >"[:80],
        ] + branch_cards + [
            "BLANK BRANCH",
            "/SWITCH",
            "C < n 1>< n 2>< Tclose  >< Top/Tde  ><   Ie     >< Vf/CLOP  ><   type   >"[:80],
        ] + switch_cards + [
            "BLANK SWITCH",
            "/SOURCE",
            "C < n 1><>< Ampl.    >< Freq.    ><Phase/T0 >< TSTART   >< TSTOP    >"[:80],
            src_a,
            src_b,
            src_c,
            "BLANK SOURCE",
            "/OUTPUT",
            "  SCA   SCB   SCC",
            "  MBA   SCA   MBB   SCB   MBC   SCC",
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
        loadloss_pct = float(tx_spec_dict.get("loadloss_pct", r_pct))

        z0_pu = float(np.sqrt((r0_pct/100.0)**2 + (x0_pct/100.0)**2))
        losses_zero_kw = (r0_pct / 100.0) * float(kvas[0])
        losses_pos_kw = (loadloss_pct / 100.0) * float(kvas[0])

        phase_v = operating_point.phase_voltages_v[target_tx]
        phase_ang = operating_point.phase_angles_deg[target_tx]

        transformer = TransformerSpec(
            name=str(tx_spec_dict["name"]),
            frequency_hz=freq_hz,
            windings=(
                TransformerWinding("HV", float(kvs[0]), float(kvas[0]) / 1000.0, tx_spec_dict.get("conns", ["D", "Y"])[0]),
                TransformerWinding("LV", float(kvs[1]), float(kvas[1]) / 1000.0, tx_spec_dict.get("conns", ["D", "Y"])[1])
            ),
            short_circuit_tests=(
                ShortCircuitTest(1, 2, z_pos_pu=float(np.sqrt((r_pct/100.0)**2 + (xhl_pct/100.0)**2)), losses_pos_kw=losses_pos_kw, test_mva=float(kvas[0])/1000.0, z_zero_pu=z0_pu, losses_zero_kw=losses_zero_kw, zero_test_mva=float(kvas[0])/1000.0),
            ),
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
