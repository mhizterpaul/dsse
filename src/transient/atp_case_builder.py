import os
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal, Any
import numpy as np

from src.transient.models import (
    TransformerSpec,
    ThreePhaseThevenin,
    TestBranch,
    SimulationConfig,
)


class ATPCaseBuilder:
    """
    Serialization layer for generating valid ATP-EMTP circuit cases from reduced domain models:
    Upstream Thévenin -> BCTRAN Transformer -> Downstream Base LV Network Thévenin -> Test Branches/Switches.

    Remove legacy feeder line models, load models, and scalar R-L transformer calculations.
    """

    def __init__(self, template_path: Optional[str] = None):
        self.template_path = template_path

    def build_explicit(
        self,
        transformer: TransformerSpec,
        upstream: ThreePhaseThevenin,
        downstream: ThreePhaseThevenin,
        events: List[TestBranch],
        simulation: SimulationConfig,
        bctran_punch: str,
        output_path: Optional[str] = None,
        scenario_id: str = "transient_scenario",
    ) -> str:
        """
        Generates a valid ATP case file from reduced multi-port Thévenin equivalents,
        BCTRAN punched coupled transformer matrix, and explicit test event branches.
        """
        if not transformer:
            raise ValueError("TransformerSpec must be provided to ATPCaseBuilder")
        if not upstream:
            raise ValueError("Upstream ThreePhaseThevenin must be provided to ATPCaseBuilder")
        if not downstream:
            raise ValueError("Downstream ThreePhaseThevenin must be provided to ATPCaseBuilder")
        if events is None:
            raise ValueError("Events list must be provided to ATPCaseBuilder")
        if not simulation:
            raise ValueError("SimulationConfig must be provided to ATPCaseBuilder")
        if not bctran_punch:
            raise ValueError("BCTRAN punch matrix string must be provided to ATPCaseBuilder")

        freq_hz = float(transformer.frequency_hz)

        def _type14_source(
            node: str, amplitude: float, frequency: float, phase_deg: float, t_start: float = 0.0, t_stop: float = 1000.0
        ) -> str:
            n_s = f"{node:<6}"[:6]
            flag_s = " 0"
            a_s = f"{amplitude:10.3f}"[:10]
            f_s = f"{frequency:10.3f}"[:10]
            p_s = f"{phase_deg:10.3f}"[:10]
            a1_s = f"{-1.0:10.3f}"[:10]
            t1_s = f"{1000.0:10.3f}"[:10]
            t0_s = f"{max(t_start, 0.0):10.3f}"[:10]
            t1_end_s = f"{t_stop:10.3f}"[:10]
            return f"14{n_s}{flag_s}{a_s}{f_s}{p_s}{a1_s}{t1_s}{t0_s}{t1_end_s}"

        def fmt_branch(n1: str, n2: str, r: float, l_mH: float, c_uF: float) -> str:
            n1_s = f"{n1:<6}"[:6]
            n2_s = f"{n2:<6}"[:6]
            r_s = f"{r:10.4f}"
            l_s = f"{l_mH:10.4f}"
            c_s = f"{c_uF:10.4f}"
            return f"  {n1_s}{n2_s}{'':<12}{r_s[:10]:>10}{l_s[:10]:>10}{c_s[:10]:>10}".rstrip()

        def fmt_switch(
            n1: str,
            n2: str,
            t_close: float,
            t_open: float,
            i_initial: float = 0.0,
            vf: float = 0.0,
            switch_type: int = 0,
        ) -> str:
            n1_s = f"{n1:<6}"[:6]
            n2_s = f"{n2:<6}"[:6]
            return (
                f"  {n1_s}{n2_s}"
                f"{t_close:10.4f}"
                f"{t_open:10.4f}"
                f"{i_initial:10.4f}"
                f"{vf:10.4f}"
                f"{switch_type:10d}"
            )

        # 1. Upstream Type-14 Sources (peak amplitude and angle from upstream.v_th)
        amp_up = np.abs(upstream.v_th) * np.sqrt(2.0)
        ang_up = np.rad2deg(np.angle(upstream.v_th))

        src_up_a = _type14_source("SRCA", amp_up[0], freq_hz, ang_up[0], 0.0, 1.0e3)
        src_up_b = _type14_source("SRCB", amp_up[1], freq_hz, ang_up[1], 0.0, 1.0e3)
        src_up_c = _type14_source("SRCC", amp_up[2], freq_hz, ang_up[2], 0.0, 1.0e3)

        branch_cards = []
        switch_cards = []

        # High-resistance grounding paths for sources
        for ph_char in ["A", "B", "C"]:
            branch_cards.append(fmt_branch(f"SRC{ph_char}", "", 1e8, 0.0, 0.0))

        # Upstream Thévenin Z_th_HV Branches (SRCA, B, C to HVA, HVB, HVC)
        # BCTRAN punch uses 3-phase nodes HVA, HVB, HVC and LVA, LVB, LVC
        ph_chars = ["A", "B", "C"]
        for i, ph in enumerate(ph_chars):
            r_hv = max(float(np.real(upstream.z_th[i, i])), 1e-4)
            x_hv = max(float(np.imag(upstream.z_th[i, i])), 1e-4)
            l_hv_mH = (x_hv / (2.0 * np.pi * freq_hz)) * 1000.0
            branch_cards.append(fmt_branch(f"SRC{ph}", f"HV{ph}", r_hv, l_hv_mH, 0.0))

        # 2. Downstream Base LV Network Active Thévenin (VTHA, VTHB, VTHC and Z_th_LV connecting to LVA, LVB, LVC)
        amp_dn = np.abs(downstream.v_th) * np.sqrt(2.0)
        ang_dn = np.rad2deg(np.angle(downstream.v_th))

        src_dn_a = _type14_source("VTHA", amp_dn[0], freq_hz, ang_dn[0], 0.0, 1.0e3)
        src_dn_b = _type14_source("VTHB", amp_dn[1], freq_hz, ang_dn[1], 0.0, 1.0e3)
        src_dn_c = _type14_source("VTHC", amp_dn[2], freq_hz, ang_dn[2], 0.0, 1.0e3)

        for i, ph in enumerate(ph_chars):
            branch_cards.append(fmt_branch(f"VTH{ph}", "", 1e8, 0.0, 0.0))
            r_lv = max(float(np.real(downstream.z_th[i, i])), 1e-4)
            x_lv = max(float(np.imag(downstream.z_th[i, i])), 1e-4)
            l_lv_mH = (x_lv / (2.0 * np.pi * freq_hz)) * 1000.0
            branch_cards.append(fmt_branch(f"LV{ph}", f"VTH{ph}", r_lv, l_lv_mH, 0.0))

        # 3. Test Event Branches & Switches attached at transformer LV port (LVA, LVB, LVC)
        for idx, ev in enumerate(events):
            t_close = float(ev.start_time_s)
            t_open = float(ev.end_time_s)

            if ev.branch_type in ["equipment", "load"]:
                r_eq = float(ev.model.get("R", 1.0))
                l_mH = float(ev.model.get("L_mH", 0.0))
                c_uF = float(ev.model.get("C_uF", 0.0))

                node_prefix = f"E{idx}"
                for p_idx in ev.phases:
                    ph = ph_chars[p_idx]
                    sec_node = f"LV{ph}"
                    load_node = f"{node_prefix}{ph}"
                    branch_cards.append(fmt_branch(load_node, "", r_eq, l_mH, c_uF))
                    switch_cards.append(fmt_switch(sec_node, load_node, t_close, t_open))

            elif ev.branch_type in ["fault"]:
                f_res = float(ev.model.get("fault_resistance_ohm", 0.001))
                f_type = str(ev.model.get("fault_type", "LG"))

                if f_type in ["LG", "LLG", "LLL"]:
                    for p_idx in ev.phases:
                        ph = ph_chars[p_idx]
                        sec_node = f"LV{ph}"
                        fault_node = f"F{idx}_{ph}"
                        branch_cards.append(fmt_branch(fault_node, "", f_res, 0.0, 0.0))
                        switch_cards.append(fmt_switch(sec_node, fault_node, t_close, t_open))
                elif f_type in ["LL"]:
                    # Phase-to-phase fault between faulted phases
                    if len(ev.phases) >= 2:
                        p1, p2 = ev.phases[0], ev.phases[1]
                        node1 = f"LV{ph_chars[p1]}"
                        node2 = f"LV{ph_chars[p2]}"
                        f_node = f"F{idx}_LL"
                        branch_cards.append(fmt_branch(f_node, "", f_res, 0.0, 0.0))
                        switch_cards.append(fmt_switch(node1, f_node, t_close, t_open))
                        switch_cards.append(fmt_switch(node2, f_node, t_close, t_open))

        atp_lines = [
            "BEGIN NEW DATA CASE",
            f"C  ATP Case File for {scenario_id}",
            f"POWER FREQUENCY                      {freq_hz:.0f}.",
            "$DUMMY, XYZ000",
            "C  dT  >< Tmax >< Xopt >< Copt ><Epsiln>",
            f"{simulation.time_step_s:8.2E}{simulation.t_stop_s:8.2E}      0.      0.",
            "    1000       1       1       1       1       0       0       1       0",
            "/BRANCH",
            "C < n1 >< n2 ><ref1><ref2>< R  >< L  >< C  >",
        ] + branch_cards + [
            "C --- BCTRAN Transformer Punch Matrix ---",
            bctran_punch,
            "/SWITCH",
            "C < n 1>< n 2>< Tclose ><Top/Tde ><   Ie   ><Vf/CLOP ><  type  >",
        ] + switch_cards + [
            "/SOURCE",
            "C < n 1><>< Ampl.  >< Freq.  ><Phase/T0><   A1   ><   T1   >< TSTART >< TSTOP  >",
            src_up_a,
            src_up_b,
            src_up_c,
            src_dn_a,
            src_dn_b,
            src_dn_c,
            "/OUTPUT",
            "  LVA   LVB   LVC",
            "BLANK BRANCH",
            "BLANK SWITCH",
            "BLANK SOURCE",
            "BLANK OUTPUT",
            "BLANK PLOT",
            "BEGIN NEW DATA CASE",
            "BLANK",
        ]

        atp_content = "\n".join(atp_lines)
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
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

        z0_pu = float(np.sqrt((r0_pct / 100.0) ** 2 + (x0_pct / 100.0) ** 2))
        losses_zero_kw = (r0_pct / 100.0) * float(kvas[0])

        from src.transient.models import TransformerWinding, ShortCircuitTest

        transformer = TransformerSpec(
            name=str(tx_spec_dict["name"]),
            frequency_hz=freq_hz,
            windings=[
                TransformerWinding("HV", float(kvs[0]), float(kvas[0]) / 1000.0, "Delta", 0.0),
                TransformerWinding("LV", float(kvs[1]), float(kvas[1]) / 1000.0, "Wye", -30.0),
            ],
            short_circuit_tests=[
                ShortCircuitTest(
                    1,
                    2,
                    z_pos_pu=float(np.sqrt((r_pct / 100.0) ** 2 + (xhl_pct / 100.0) ** 2)),
                    losses_pos_kw=(r_pct / 100.0) * float(kvas[0]),
                    z_zero_pu=z0_pu,
                    losses_zero_kw=losses_zero_kw,
                )
            ],
            excitation_current_percent=imag_pct,
            excitation_loss_kw=max((noloadloss_pct / 100.0) * float(kvas[0]), 25.0),
        )

        from src.transient.bctran_generator import BCTRANGenerator

        punch = BCTRANGenerator().generate(transformer)

        target_tx = getattr(event, "target", "trans1")
        phase_v = operating_point.phase_voltages_v[target_tx]
        phase_ang = operating_point.phase_angles_deg[target_tx]

        v_pre = np.array(
            [
                phase_v[0] * np.exp(1j * np.deg2rad(phase_ang[0])),
                phase_v[1] * np.exp(1j * np.deg2rad(phase_ang[1])),
                phase_v[2] * np.exp(1j * np.deg2rad(phase_ang[2])),
            ],
            dtype=complex,
        )

        # Default diagonal Thévenin equivalents for legacy call
        z_hv = np.diag([0.01 + 0.05j, 0.01 + 0.05j, 0.01 + 0.05j])
        z_lv = np.diag([0.002 + 0.01j, 0.002 + 0.01j, 0.002 + 0.01j])

        upstream = ThreePhaseThevenin(
            v_th=v_pre * 80.0, z_th=z_hv, v_pre=v_pre * 80.0, i_pre=np.zeros(3, dtype=complex), frequency_hz=freq_hz
        )
        downstream = ThreePhaseThevenin(
            v_th=v_pre, z_th=z_lv, v_pre=v_pre, i_pre=np.zeros(3, dtype=complex), frequency_hz=freq_hz
        )

        if hasattr(event, "to_test_branches"):
            branches = event.to_test_branches(freq_hz)
        else:
            branches = []

        sim_config = SimulationConfig(t_start_s=0.0, t_stop_s=0.15, time_step_s=1e-4)

        return self.build_explicit(
            transformer=transformer,
            upstream=upstream,
            downstream=downstream,
            events=branches,
            simulation=sim_config,
            bctran_punch=punch,
            output_path=output_path,
            scenario_id=scenario_id,
        )
