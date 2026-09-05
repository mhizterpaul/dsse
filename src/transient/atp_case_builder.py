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
        scenario_id: str,
        output_path: Optional[str] = None,
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
        if not scenario_id:
            raise ValueError("scenario_id must be provided to ATPCaseBuilder")

        freq_hz = float(transformer.frequency_hz)

        def _type14_source(
            node: str, amplitude: float, frequency: float, phase_deg: float, t_start: float, t_stop: float
        ) -> str:
            n_s = f"{node:<6}"[:6]
            flag_s = " 0"
            a_s = f"{amplitude:10.3f}"[:10]
            f_s = f"{frequency:10.3f}"[:10]
            p_s = f"{phase_deg:10.3f}"[:10]
            a1_s = f"{'':10s}"[:10]
            t1_s = f"{'':10s}"[:10]
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

        ph_chars = ["A", "B", "C"]

        src_up_a = _type14_source("SRCA", amp_up[0], freq_hz, ang_up[0], 0.0, 1.0e3)
        src_up_b = _type14_source("SRCB", amp_up[1], freq_hz, ang_up[1], 0.0, 1.0e3)
        src_up_c = _type14_source("SRCC", amp_up[2], freq_hz, ang_up[2], 0.0, 1.0e3)

        branch_cards = []
        switch_cards = []

        # High-resistance grounding paths for sources
        for ph in ph_chars:
            branch_cards.append(fmt_branch(f"SRC{ph}", "", 1e8, 0.0, 0.0))

        # Upstream Thévenin Z_th_HV Branches (SRCA, B, C to HV_A, HV_B, HV_C)
        for i, ph in enumerate(ph_chars):
            r_hv = float(np.real(upstream.z_th[i, i]))
            x_hv = float(np.imag(upstream.z_th[i, i]))
            if r_hv <= 0.0 or x_hv <= 0.0:
                raise ValueError(f"Upstream Z_th[{i},{i}] must have positive resistance and reactance, got R={r_hv}, X={x_hv}")
            l_hv_mH = (x_hv / (2.0 * np.pi * freq_hz)) * 1000.0
            branch_cards.append(fmt_branch(f"SRC{ph}", f"HV_{ph}", r_hv, l_hv_mH, 0.0))

        # 2. Downstream Base LV Network Load Impedance Branches (LV_A, LV_B, LV_C to Ground reference '0')
        for i, ph in enumerate(ph_chars):
            r_lv = float(np.real(downstream.z_th[i, i]))
            x_lv = float(np.imag(downstream.z_th[i, i]))
            if r_lv <= 0.0 or x_lv <= 0.0:
                raise ValueError(f"Downstream Z_th[{i},{i}] must have positive resistance and reactance, got R={r_lv}, X={x_lv}")
            l_lv_mH = (x_lv / (2.0 * np.pi * freq_hz)) * 1000.0
            branch_cards.append(fmt_branch(f"LV_{ph}", "", r_lv, l_lv_mH, 0.0))

        # 3. Test Event Branches & Switches attached at transformer LV port (LV_A, LV_B, LV_C)
        for idx, ev in enumerate(events):
            t_close = float(ev.start_time_s)
            t_open = float(ev.end_time_s)

            if ev.branch_type in ["equipment", "load"]:
                if "R" not in ev.model:
                    raise ValueError(f"TestBranch '{ev.name}' missing required 'R' in model dict")
                r_eq = float(ev.model["R"])
                l_mH = float(ev.model["L_mH"]) if "L_mH" in ev.model else 0.0
                c_uF = float(ev.model["C_uF"]) if "C_uF" in ev.model else 0.0

                node_prefix = f"E{idx}"
                for p_idx in ev.phases:
                    ph = ph_chars[p_idx]
                    sec_node = f"LV_{ph}"
                    load_node = f"{node_prefix}{ph}"
                    branch_cards.append(fmt_branch(load_node, "", r_eq, l_mH, c_uF))
                    switch_cards.append(fmt_switch(sec_node, load_node, t_close, t_open))

            elif ev.branch_type in ["fault"]:
                if "fault_resistance_ohm" not in ev.model or "fault_type" not in ev.model:
                    raise ValueError(f"Fault TestBranch '{ev.name}' missing 'fault_resistance_ohm' or 'fault_type'")
                f_res = float(ev.model["fault_resistance_ohm"])
                f_type = str(ev.model["fault_type"])

                if f_type in ["LG", "LLG", "LLL"]:
                    for p_idx in ev.phases:
                        ph = ph_chars[p_idx]
                        sec_node = f"LV_{ph}"
                        fault_node = f"F{idx}_{ph}"
                        branch_cards.append(fmt_branch(fault_node, "", f_res, 0.0, 0.0))
                        switch_cards.append(fmt_switch(sec_node, fault_node, t_close, t_open))
                elif f_type in ["LL"]:
                    if len(ev.phases) >= 2:
                        p1, p2 = ev.phases[0], ev.phases[1]
                        node1 = f"LV_{ph_chars[p1]}"
                        node2 = f"LV_{ph_chars[p2]}"
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
            "/OUTPUT",
            "  LV_A  LV_B  LV_C",
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
