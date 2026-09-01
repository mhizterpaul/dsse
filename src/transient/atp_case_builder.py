import os
import traceback
from src.loads import get_equipment_model

class ATPCaseBuilder:
    def __init__(self, template_path: str = None):
        self.template_path = template_path

    def build(self, realization, operating_point, event, output_path: str) -> str:
        """
        Generates a valid ATP-EMTP card file for single equipment switching,
        explicit line faults (LG, LL, LLG, LLL), and co-events in ATP-EMTP syntax.
        """
        if realization is None or not hasattr(realization, "scenario_id"):
            raise ValueError("Realization must be provided with scenario_id attribute")
        scenario_id = realization.scenario_id

        if event is None:
            raise ValueError("Event must be provided to ATPCaseBuilder")

        # Unwrap co-events or single events
        if hasattr(event, "event_1") and hasattr(event, "event_2"):
            events_to_process = [event.event_1, event.event_2]
        else:
            events_to_process = [event]

        if not hasattr(realization, "line_parameters") or "mult" not in realization.line_parameters:
            raise ValueError("Realization missing line_parameters['mult']")
        line_mult = float(realization.line_parameters["mult"])
        R_base = 0.45 * line_mult
        L_base = 0.15 * line_mult

        # Extract target transformer 3-phase voltages and phase angles solved by OpenDSS
        if hasattr(event, "target"):
            target_tx = str(event.target)
        elif hasattr(event, "event_1") and hasattr(event.event_1, "target"):
            target_tx = str(event.event_1.target)
        else:
            raise ValueError(f"Event object '{type(event).__name__}' missing 'target' attribute")

        if not operating_point:
            raise ValueError("Operating point must be provided to ATPCaseBuilder")

        if not hasattr(operating_point, "phase_voltages_v") or target_tx not in operating_point.phase_voltages_v:
            raise ValueError(f"Operating point missing phase_voltages_v for target transformer '{target_tx}'")
        if not hasattr(operating_point, "phase_angles_deg") or target_tx not in operating_point.phase_angles_deg:
            raise ValueError(f"Operating point missing phase_angles_deg for target transformer '{target_tx}'")

        phase_v = operating_point.phase_voltages_v[target_tx]
        phase_ang = operating_point.phase_angles_deg[target_tx]

        # Convert RMS voltage to peak amplitude in Volts
        import numpy as np
        amp_a = float(phase_v[0]) * np.sqrt(2.0)
        amp_b = float(phase_v[1]) * np.sqrt(2.0)
        amp_c = float(phase_v[2]) * np.sqrt(2.0)

        ang_a = float(phase_ang[0])
        ang_b = float(phase_ang[1])
        ang_c = float(phase_ang[2])

        if operating_point is None or not hasattr(operating_point, "frequency_hz"):
            print(f"ERROR: Operating point is missing or frequency_hz attribute absent\n{traceback.format_exc()}")
            raise ValueError("Operating point must be provided with frequency_hz")

        freq_hz = float(operating_point.frequency_hz)
        if freq_hz <= 0:
            print(f"ERROR: Invalid frequency_hz in operating point: {freq_hz}\n{traceback.format_exc()}")
            raise ValueError(f"frequency_hz in operating point must be positive, got {freq_hz}")

        freq_str = f"{freq_hz:.2f}".rjust(10)
        a1_str = " ".rjust(10)
        t1_str = " ".rjust(10)
        tstart_str = f"-1.00".rjust(10)
        tstop_str = f"100.00".rjust(10)

        src_a = f"14SRCA  -1{amp_a:10.2f}{freq_str}{ang_a:10.2f}{a1_str}{t1_str}{tstart_str}{tstop_str}"
        src_b = f"14SRCB  -1{amp_b:10.2f}{freq_str}{ang_b:10.2f}{a1_str}{t1_str}{tstart_str}{tstop_str}"
        src_c = f"14SRCC  -1{amp_c:10.2f}{freq_str}{ang_c:10.2f}{a1_str}{t1_str}{tstart_str}{tstop_str}"

        branch_cards = []
        switch_cards = []

        # Default high-resistance paths to ground
        branch_cards.extend([
            "  SRCA                      1.E8                                               0",
            "  SRCB                      1.E8                                               0",
            "  SRCC                      1.E8                                               0",
        ])

        # Explicit Feeder Line Parameters
        r_line = float(realization.line_parameters.get("r1", 0.21))
        x_line = float(realization.line_parameters.get("x1", 0.08))
        l_line_mH = (x_line / (2 * 3.14159 * freq_hz)) * 1000.0

        r_line_str = f"{r_line:.4f}".rjust(10)
        l_line_str = f"{l_line_mH:.4f}".rjust(10)

        for ph_char in ["A", "B", "C"]:
            src_node = f"SRC{ph_char}"
            tx_node = f"TX_{ph_char}"
            branch_cards.append(f"  {src_node}  {tx_node}             {r_line_str}{l_line_str}                                               0")

        # Explicit BCTRAN Transformer Model Parameters
        tx_spec = getattr(realization, "transformer_spec", {})
        r_tx_pct = float(tx_spec.get("r_pct", 0.60))
        x_tx_pct = float(tx_spec.get("xhl_pct", 8.33))
        kva = float(tx_spec.get("kvas", [1500.0])[0])
        kv_lv = float(tx_spec.get("kvs", [11.0, 0.415])[1])

        z_base = (kv_lv ** 2 * 1000.0) / kva
        r_tx = (r_tx_pct / 100.0) * z_base
        x_tx = (x_tx_pct / 100.0) * z_base
        l_tx_mH = (x_tx / (2 * 3.14159 * freq_hz)) * 1000.0

        r_tx_str = f"{r_tx:.4f}".rjust(10)
        l_tx_str = f"{l_tx_mH:.4f}".rjust(10)

        for ph_char in ["A", "B", "C"]:
            tx_node = f"TX_{ph_char}"
            sec_node = f"SEC{ph_char}"
            branch_cards.append(f"  {tx_node}  {sec_node}             {r_tx_str}{l_tx_str}                                               0")

        # Equipment switching and fault events
        for idx, ev in enumerate(events_to_process):
            if not hasattr(ev, "start_time_s"):
                raise ValueError(f"Event object '{type(ev).__name__}' missing 'start_time_s' attribute")
            start_s = float(ev.start_time_s)
            duration_s = float(getattr(ev, "duration_s", 0.5))
            end_s = start_s + duration_s

            start_str = f"{start_s:.4f}".rjust(10)
            end_str = f"{end_s:.4f}".rjust(10)

            if not hasattr(ev, "event_class"):
                raise ValueError(f"Event object '{type(ev).__name__}' missing 'event_class' attribute")
            ev_class = str(ev.event_class)

            if ev_class == "equipment_switch":
                eq_type = str(getattr(ev, "equipment_type"))
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
                    raise ValueError(f"Equipment model {eq_type} missing R or X atp_params")

                r_eq = float(r_stator)
                x_eq = float(x_stator)

                r_str = f"{r_eq:.4f}".rjust(10)
                l_str = f"{x_eq * 1000.0 / (2*3.14159*freq_hz):.4f}".rjust(10)

                node_prefix = f"E{idx}"
                for ph_char in ["A", "B", "C"]:
                    sec_node = f"SEC{ph_char}"
                    load_node = f"{node_prefix}{ph_char}"
                    branch_cards.append(f"  {load_node}                       {r_str}{l_str}                                               0")
                    switch_cards.append(f"  {sec_node}  {load_node}       {start_str}{end_str}                                             0")

            elif ev_class == "line_fault":
                f_phases = getattr(ev, "faulted_phases", (0,))
                f_res = float(getattr(ev, "fault_resistance", 0.001))
                r_fault_str = f"{f_res:.4f}".rjust(10)

                ph_chars = ["A", "B", "C"]
                for p_idx in f_phases:
                    ph_char = ph_chars[p_idx]
                    sec_node = f"SEC{ph_char}"
                    fault_node = f"FLT{idx}{ph_char}"
                    branch_cards.append(f"  {fault_node}                       {r_fault_str}                                               0")
                    switch_cards.append(f"  {sec_node}  {fault_node}       {start_str}{end_str}                                             0")

        if not switch_cards:
            start_str = f"0.0200".rjust(10)
            branch_cards.extend([
                f"  S0A                       0.5000   10.0000    0.8000                                     0",
                f"  S0B                       0.5000   10.0000    0.8000                                     0",
                f"  S0C                       0.5000   10.0000    0.8000                                     0",
            ])
            switch_cards.extend([
                f"  SRCA  S0A       {start_str}      1.E3                                             0",
                f"  SRCB  S0B       {start_str}      1.E3                                             0",
                f"  SRCC  S0C       {start_str}      1.E3                                             0",
            ])

        atp_lines = [
            "BEGIN NEW DATA CASE",
            f"C  ATP Case File for {scenario_id}",
            f"POWER FREQUENCY                      {freq_hz:.0f}.",
            "$DUMMY, XYZ000",
            "C  dT  >< Tmax >< Xopt >< Copt ><Epsiln>",
            "   1.E-4    0.1     50.     50.",
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
