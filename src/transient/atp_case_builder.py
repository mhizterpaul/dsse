import os
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal, Any
import numpy as np


@dataclass(frozen=True)
class TransformerWinding:
    name: str
    rated_kv: float  # Line-to-line voltage rating in kV
    rated_mva: float
    connection: str = "Y"  # "Y", "D", "A"
    phase_shift_deg: float = 0.0
    grounded: bool = True


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
    vector_group: str = "Yy0"


@dataclass(frozen=True)
class ThreePhaseThevenin:
    name: str
    v_th: Tuple[complex, complex, complex]  # RMS complex phasors
    z_th: np.ndarray                        # 3x3 complex impedance matrix in Ohms
    v_pre: Tuple[complex, complex, complex] = (0j, 0j, 0j)
    i_pre: Tuple[complex, complex, complex] = (0j, 0j, 0j)


@dataclass(frozen=True)
class NetworkEquivalent:
    name: str
    side: Literal["hv", "lv"]
    vth_v: Tuple[complex, complex, complex]
    zth_ohm: np.ndarray
    reference_bus: str = ""


@dataclass(frozen=True)
class TestBranchModel:
    name: str
    from_bus: str
    to_bus: str
    r_ohm: float
    l_h: float
    c_f: float = 0.0
    initial_closed: bool = True


@dataclass(frozen=True)
class ThreePhaseState:
    voltage_rms_v: Tuple[float, float, float]
    voltage_angle_deg: Tuple[float, float, float]


@dataclass(frozen=True)
class SourceModel:
    name: str
    bus: str
    frequency_hz: float
    voltage_rms_v: Tuple[float, float, float]
    voltage_angle_deg: Tuple[float, float, float]


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
    t_start_s: float
    t_stop_s: float
    time_step_s: float


def bctran_vrat_kv(rated_kv_ll: float, connection: str) -> float:
    conn = connection.upper()
    if conn in {"Y", "WYE"}:
        return rated_kv_ll / np.sqrt(3.0)
    if conn in {"D", "DELTA"}:
        return rated_kv_ll
    raise ValueError(f"Unsupported transformer connection: {connection}")


class BCTRANBuilder:
    """
    Builder for BCTRAN transformer supporting routine cards and coupled transformer representations.
    """

    def build_input(self, transformer: TransformerSpec, node_map: Optional[dict] = None) -> str:
        if node_map is None:
            node_map = {
                "hv_a": "HV_A", "hv_b": "HV_B", "hv_c": "HV_C",
                "lv_a": "LV_A", "lv_b": "LV_B", "lv_c": "LV_C",
            }

        hv = transformer.windings[0]
        lv = transformer.windings[1]
        sc = transformer.short_circuit_tests[0]

        freq = transformer.frequency_hz
        iexpos = transformer.excitation_current_percent if transformer.excitation_current_percent is not None else 0.8
        spos_mva = hv.rated_mva
        lexpos_kw = transformer.excitation_loss_kw if transformer.excitation_loss_kw is not None else (0.001 * spos_mva * 1000.0)

        hv_vrat = bctran_vrat_kv(hv.rated_kv, hv.connection)
        lv_vrat = bctran_vrat_kv(lv.rated_kv, lv.connection)

        zpos_pct = 100.0 * sc.z_pos_pu
        zzero_pct = 100.0 * (sc.z_zero_pu if sc.z_zero_pu is not None else sc.z_pos_pu)
        lpos_kw = sc.losses_pos_kw
        lzero_kw = sc.losses_zero_kw if sc.losses_zero_kw is not None else sc.losses_pos_kw

        exct_card = f" 2 {freq:6.2f}{iexpos:8.4f}{spos_mva:8.2f}{lexpos_kw:8.2f}{iexpos:8.4f}{spos_mva:8.2f}{lexpos_kw:8.2f} 0 2 2"

        hv_nodes = f"{node_map['hv_a']:<6}{node_map['hv_b']:<6}{node_map['hv_c']:<6}"
        wind_card_1 = f" 1 {hv_vrat:10.4f} 0.        {hv_nodes}"

        lv_nodes = f"{node_map['lv_a']:<6}{node_map['lv_b']:<6}{node_map['lv_c']:<6}"
        wind_card_2 = f" 2 {lv_vrat:10.4f} 0.        {lv_nodes}"

        sc_card = f" 1 2 {lpos_kw:8.2f}{zpos_pct:8.4f}{spos_mva:8.2f}{lzero_kw:8.2f}{zzero_pct:8.4f}{spos_mva:8.2f} 0 1"

        return "\n".join([
            "BEGIN NEW DATA CASE",
            "ACCESS MODULE BCTRAN",
            "$ERASE",
            "C Excitation test data",
            exct_card,
            "C Winding data",
            wind_card_1,
            wind_card_2,
            "C Short circuit test data",
            sc_card,
            "BLANK",
            "$PUNCH",
            "BLANK",
            "BEGIN NEW DATA CASE",
            "BLANK"
        ])

    def build_branch_cards(self, transformer: TransformerSpec, node_map: Optional[dict] = None) -> List[str]:
        """
        Builds standard 3-phase distribution transformer branch representation with correct LV impedance.
        """
        if node_map is None:
            node_map = {
                "hv_a": "HV_A", "hv_b": "HV_B", "hv_c": "HV_C",
                "lv_a": "LV_A", "lv_b": "LV_B", "lv_c": "LV_C",
            }

        hv = transformer.windings[0]
        lv = transformer.windings[1]
        sc = transformer.short_circuit_tests[0]

        freq = transformer.frequency_hz
        mva = hv.rated_mva if hv.rated_mva > 0 else 1.5
        v_lv_kv = lv.rated_kv

        z_base_lv = (v_lv_kv ** 2) / mva

        z_pu = sc.z_pos_pu
        r_pu = sc.losses_pos_kw / (mva * 1000.0) if mva > 0 else 0.01
        x_pu = np.sqrt(max(0.0, z_pu ** 2 - r_pu ** 2))

        r_lv = r_pu * z_base_lv
        x_lv = x_pu * z_base_lv
        l_lv_mH = (x_lv / (2.0 * np.pi * freq)) * 1000.0

        def fmt_branch(n1: str, n2: str, r: float, l_mH: float) -> str:
            n1_s = f"{n1:<6}"[:6]
            n2_s = f"{n2:<6}"[:6]
            r_s = f"{r:10.4f}"
            l_s = f"{l_mH:10.4f}"
            return f"  {n1_s}{n2_s}{'':<12}{r_s[:10]:>10}{l_s[:10]:>10}{'0.0':>10}".rstrip()

        cards = [
            "C BCTRAN Transformer Model (LV Equivalent Series Branch)",
            fmt_branch(node_map["hv_a"], node_map["lv_a"], r_lv, l_lv_mH),
            fmt_branch(node_map["hv_b"], node_map["lv_b"], r_lv, l_lv_mH),
            fmt_branch(node_map["hv_c"], node_map["lv_c"], r_lv, l_lv_mH),
        ]
        return cards


class ATPCaseBuilder:
    def __init__(self, template_path: str = None):
        self.template_path = template_path
        self.bctran_builder = BCTRANBuilder()

    def build_explicit(
        self,
        transformer: TransformerSpec,
        upstream: ThreePhaseThevenin,
        downstream: ThreePhaseThevenin,
        event: TransientEvent,
        simulation: SimulationConfig,
        test_branches: Optional[List[TestBranchModel]] = None,
        output_path: Optional[str] = None,
        scenario_id: str = "transient_scenario"
    ) -> str:
        """
        Generates a valid ATP-EMTP card file for reduced-order two-port Thévenin EMT transients:
        Upstream Thévenin (HV) -> BCTRAN Transformer -> Downstream Base LV Thévenin // Test Branch(es)/Fault.
        """
        if transformer is None:
            raise ValueError("TransformerSpec must be provided")
        if upstream is None:
            raise ValueError("Upstream ThreePhaseThevenin must be provided")
        if downstream is None:
            raise ValueError("Downstream ThreePhaseThevenin must be provided")
        if event is None:
            raise ValueError("TransientEvent must be provided")
        if simulation is None:
            raise ValueError("SimulationConfig must be provided")

        freq_hz = transformer.frequency_hz

        def _type14_source(node: str, v_complex: complex, frequency: float, t_start: float, t_stop: float) -> str:
            amp = np.abs(v_complex) * np.sqrt(2.0)
            phase_deg = np.degrees(np.angle(v_complex))
            n_s = f"{node:<6}"[:6]
            return f"14{n_s}{'0':>6}{amp:>10.3f}{frequency:>10.3f}{phase_deg:>10.3f}{t_start:>10.3f}{t_stop:>10.3f}"

        src_hv_a = _type14_source("SRC_HA", upstream.v_th[0], freq_hz, -1.0, 1.0e3)
        src_hv_b = _type14_source("SRC_HB", upstream.v_th[1], freq_hz, -1.0, 1.0e3)
        src_hv_c = _type14_source("SRC_HC", upstream.v_th[2], freq_hz, -1.0, 1.0e3)

        branch_cards = []
        switch_cards = []

        def fmt_branch(n1: str, n2: str, r: float, l_mH: float, c_uF: float) -> str:
            n1_s = f"{n1:<6}"[:6]
            n2_s = f"{n2:<6}"[:6]
            r_s = f"{r:10.4f}"
            l_s = f"{l_mH:10.4f}"
            c_s = f"{c_uF:10.4f}"
            return f"  {n1_s}{n2_s}{'':<12}{r_s[:10]:>10}{l_s[:10]:>10}{c_s[:10]:>10}".rstrip()

        def fmt_switch(n1: str, n2: str, t_close: float, t_open: float) -> str:
            n1_s = f"{n1:<6}"[:6]
            n2_s = f"{n2:<6}"[:6]
            tc_s = f"{t_close:10.4f}"
            to_s = f"{t_open:10.4f}"
            return f"  {n1_s}{n2_s}{tc_s}{to_s}"

        # 1. Upstream HV Impedance Zth_HV
        for idx, ph in enumerate(["A", "B", "C"]):
            src_node = f"SRC_H{ph}"
            hv_node = f"HV_{ph}"
            r_h = float(np.real(upstream.z_th[idx, idx]))
            x_h = float(np.imag(upstream.z_th[idx, idx]))
            l_h_mH = (x_h / (2.0 * np.pi * freq_hz)) * 1000.0 if freq_hz > 0 else 0.0
            branch_cards.append(fmt_branch(src_node, hv_node, r_h, l_h_mH, 0.0))

        # 2. BCTRAN Transformer Branch Model
        tx_branch_cards = self.bctran_builder.build_branch_cards(transformer)
        branch_cards.extend(tx_branch_cards)

        # 3. Downstream Base LV Thévenin Equivalent (Shunt impedance Zth_LV at TEST BUS / LV)
        has_vth_lv = any(np.abs(v) > 1e-3 for v in downstream.v_th)
        source_cards = [src_hv_a, src_hv_b, src_hv_c]

        for idx, ph in enumerate(["A", "B", "C"]):
            lv_node = f"LV_{ph}"
            r_l = float(np.real(downstream.z_th[idx, idx]))
            x_l = float(np.imag(downstream.z_th[idx, idx]))
            l_l_mH = (x_l / (2.0 * np.pi * freq_hz)) * 1000.0 if freq_hz > 0 else 0.0

            if has_vth_lv:
                src_lv_node = f"SRC_L{ph}"
                branch_cards.append(fmt_branch(lv_node, src_lv_node, r_l, l_l_mH, 0.0))
                source_cards.append(_type14_source(src_lv_node, downstream.v_th[idx], freq_hz, -1.0, 1.0e3))
            else:
                branch_cards.append(fmt_branch(lv_node, "", r_l, l_l_mH, 0.0))

        # 4. Explicit Test Branches from parameters or event
        if test_branches:
            for tb in test_branches:
                l_mH = tb.l_h * 1000.0
                c_uF = tb.c_f * 1e6
                branch_cards.append(fmt_branch(tb.from_bus, tb.to_bus, tb.r_ohm, l_mH, c_uF))

        # 5. Transient Events & Switches
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

            if ev_class in ["line_fault", "fault"]:
                f_phases = getattr(ev, "faulted_phases", getattr(event, "faulted_phases"))
                f_res = float(getattr(ev, "fault_resistance_ohm", getattr(ev, "fault_resistance", getattr(event, "fault_resistance_ohm"))))
                ph_chars = ["A", "B", "C"]
                for p_idx in f_phases:
                    ph_char = ph_chars[p_idx]
                    lv_node = f"LV_{ph_char}"
                    fault_node = f"FLT_{idx}_{ph_char}"
                    branch_cards.append(fmt_branch(fault_node, "", f_res, 0.0, 0.0))
                    switch_cards.append(fmt_switch(lv_node, fault_node, start_s, end_s))
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

                if r_val is None:
                    err_msg = f"Equipment model '{eq_type}' missing required R in atp_params"
                    print(f"ERROR: {err_msg}\n{traceback.format_exc()}")
                    raise ValueError(err_msg)

                r_eq = float(r_val)
                node_prefix = f"TST_{idx}"
                for ph_char in ["A", "B", "C"]:
                    lv_node = f"LV_{ph_char}"
                    test_node = f"{node_prefix}_{ph_char}"
                    branch_cards.append(fmt_branch(test_node, "", r_eq, l_mH_val, 0.0))
                    switch_cards.append(fmt_switch(lv_node, test_node, start_s, end_s))

        atp_lines = [
            "BEGIN NEW DATA CASE",
            f"C  ATP Case File for {scenario_id}",
            f"POWER FREQUENCY                      {freq_hz:.0f}.",
            "$DUMMY, XYZ000",
            "C  dT  >< Tmax >< Xopt >< Copt ><Epsiln>",
            f"{simulation.time_step_s:10.6E}{simulation.t_stop_s:10.6E}     0.     0.",
            "    1000       1       1       1       1       0       0       1       0",
            "/BRANCH",
            "C < n1 >< n2 ><ref1><ref2>< R  >< L  >< C  >",
        ] + branch_cards + [
            "/SWITCH",
            "C < n 1>< n 2>< Tclose ><Top/Tde ><   Ie   ><Vf/CLOP ><  type  >",
        ] + switch_cards + [
            "/SOURCE",
            "C < n 1><>< Ampl.  >< Freq.  ><Phase/T0><   A1   ><   T1   >< TSTART >< TSTOP  >",
        ] + source_cards + [
            "/OUTPUT",
            "  LV_A  LV_B  LV_C",
            "BLANK BRANCH",
            "BLANK SWITCH",
            "BLANK SOURCE",
            "BLANK OUTPUT",
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

        if not operating_point:
            raise ValueError("Operating point must be provided to ATPCaseBuilder")

        freq_hz = float(getattr(operating_point, "frequency_hz"))

        tx_spec_dict = getattr(realization, "transformer_spec")
        r_pct = float(tx_spec_dict["r_pct"])
        xhl_pct = float(tx_spec_dict["xhl_pct"])
        r0_pct = float(tx_spec_dict.get("r0_pct", r_pct))
        x0_pct = float(tx_spec_dict.get("x0_pct", xhl_pct))
        kvas = tx_spec_dict["kvas"]
        kvs = tx_spec_dict["kvs"]
        noloadloss_pct = float(tx_spec_dict.get("noloadloss_pct", 0.1))
        imag_pct = float(tx_spec_dict.get("imag_pct", 0.8))

        z0_pu = float(np.sqrt((r0_pct / 100.0) ** 2 + (x0_pct / 100.0) ** 2))
        losses_zero_kw = (r0_pct / 100.0) * float(kvas[0])

        phase_v = operating_point.phase_voltages_v[target_tx]
        phase_ang = operating_point.phase_angles_deg[target_tx]

        transformer = TransformerSpec(
            name=str(tx_spec_dict["name"]),
            frequency_hz=freq_hz,
            windings=[
                TransformerWinding("HV", float(kvs[0]), float(kvas[0]) / 1000.0, "Y", 0.0),
                TransformerWinding("LV", float(kvs[1]), float(kvas[1]) / 1000.0, "Y", 0.0)
            ],
            short_circuit_tests=[
                ShortCircuitTest(1, 2, z_pos_pu=float(np.sqrt((r_pct / 100.0) ** 2 + (xhl_pct / 100.0) ** 2)), losses_pos_kw=(r_pct / 100.0) * float(kvas[0]), z_zero_pu=z0_pu, losses_zero_kw=losses_zero_kw)
            ],
            excitation_current_percent=imag_pct,
            excitation_loss_kw=(noloadloss_pct / 100.0) * float(kvas[0]),
            vector_group="Yy0"
        )

        v_hv_complex = (
            phase_v[0] * np.exp(1j * np.radians(phase_ang[0])),
            phase_v[1] * np.exp(1j * np.radians(phase_ang[1])),
            phase_v[2] * np.exp(1j * np.radians(phase_ang[2]))
        )

        upstream_th = ThreePhaseThevenin(
            name="hv_upstream_grid",
            v_th=v_hv_complex,
            z_th=np.diag([0.01 + 1j * 0.05, 0.01 + 1j * 0.05, 0.01 + 1j * 0.05])
        )

        downstream_th = ThreePhaseThevenin(
            name="lv_base_network",
            v_th=(0j, 0j, 0j),
            z_th=np.diag([10.0 + 1j * 2.0, 10.0 + 1j * 2.0, 10.0 + 1j * 2.0])
        )

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

        t_start_s = 0.0
        t_stop_s = start_s + dur_s + 0.05
        sim_config = SimulationConfig(t_start_s=t_start_s, t_stop_s=t_stop_s, time_step_s=1e-4)

        return self.build_explicit(
            transformer=transformer,
            upstream=upstream_th,
            downstream=downstream_th,
            event=transient_ev,
            simulation=sim_config,
            output_path=output_path,
            scenario_id=scenario_id
        )
