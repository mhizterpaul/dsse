import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.signal
import pywt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner
from src.transient.events import (
    SingleEquipmentSwitchEvent,
    EquipmentEquipmentCoEvent,
    NoLoadEvent,
)


def run_no_load_vs_loaded_simulation(
    equipment_type: str = "ac_motor",
    start_time_s: float = 0.02,
    duration_s: float = 0.10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Runs Case A (No-Load) and Case B (Loaded Event) in ATP under identical circuit conditions.
    """
    runner = CoSimulationRunner()
    plant_data = runner.initialize_plant_session(use_baseline_feeder=True, seed=seed)
    runner.dss.run_command("disable Fault.*")
    op = plant_data["op"] if "op" in plant_data else None
    if op is None:
        from src.power_plant.plant import solve_operating_point
        op = solve_operating_point(runner.dss)

    tx_unit_id = "trans1_lv_boundary"
    t_stop = start_time_s + duration_s + 0.03

    # Case A: No-load event
    no_load_ev = NoLoadEvent(start_time_s=start_time_s, duration_s=duration_s)
    t_no, v_no_dict, i_no_dict, _ = runner.measure_transients(
        op=op,
        event=no_load_ev,
        scenario_id="noload_case_a",
        feeder_idx=1,
        use_baseline_feeder=True,
        t_stop_override=t_stop
    )

    # Case B: Loaded event
    loaded_ev = SingleEquipmentSwitchEvent(
        equipment_type=equipment_type,
        start_time_s=start_time_s,
        duration_s=duration_s,
        target="trans1",
        parameters={}
    )
    t_ld, v_ld_dict, i_ld_dict, _ = runner.measure_transients(
        op=op,
        event=loaded_ev,
        scenario_id="loaded_case_b",
        feeder_idx=1,
        use_baseline_feeder=True,
        t_stop_override=t_stop
    )

    v_no = v_no_dict[tx_unit_id]
    i_no = i_no_dict[tx_unit_id]
    v_ld = v_ld_dict[tx_unit_id]
    i_ld = i_ld_dict[tx_unit_id]

    min_len = min(len(t_no), len(t_ld))
    t = t_ld[:min_len]
    v_no = v_no[:min_len]
    i_no = i_no[:min_len]
    v_ld = v_ld[:min_len]
    i_ld = i_ld[:min_len]

    v_event = v_ld - v_no
    i_event = i_ld - i_no

    return {
        "time": t,
        "v_no_load": v_no,
        "i_no_load": i_no,
        "v_loaded": v_ld,
        "i_loaded": i_ld,
        "v_event": v_event,
        "i_event": i_event,
        "equipment_type": equipment_type,
    }


def run_load_pair_vs_single_simulation(
    equipment_1: str = "ac_motor",
    equipment_2: str = "compressor",
    start_time_s: float = 0.02,
    duration_s: float = 0.10,
    time_offset_s: float = 0.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Runs Case A (No-Load), Case B1 (Single Load 1), Case B2 (Single Load 2), and Case C (Joint Load Pair)
    under identical circuit parameters in ATP.

    Computes single load residuals, joint load pair residual, linear composed sum, and interaction residual.
    """
    runner = CoSimulationRunner()
    plant_data = runner.initialize_plant_session(use_baseline_feeder=True, seed=seed)
    runner.dss.run_command("disable Fault.*")
    op = plant_data["op"] if "op" in plant_data else None
    if op is None:
        from src.power_plant.plant import solve_operating_point
        op = solve_operating_point(runner.dss)

    tx_unit_id = "trans1_lv_boundary"
    t_stop = start_time_s + duration_s + time_offset_s + 0.03

    # Case A: No-load event
    no_load_ev = NoLoadEvent(start_time_s=start_time_s, duration_s=duration_s + time_offset_s)
    t_no, v_no_dict, i_no_dict, _ = runner.measure_transients(
        op=op,
        event=no_load_ev,
        scenario_id="pair_noload",
        feeder_idx=1,
        use_baseline_feeder=True,
        t_stop_override=t_stop
    )

    # Case B1: Single Load 1
    ev1 = SingleEquipmentSwitchEvent(
        equipment_type=equipment_1,
        start_time_s=start_time_s,
        duration_s=duration_s,
        target="trans1",
        parameters={}
    )
    t_1, v1_dict, i1_dict, _ = runner.measure_transients(
        op=op,
        event=ev1,
        scenario_id="pair_single1",
        feeder_idx=1,
        use_baseline_feeder=True,
        t_stop_override=t_stop
    )

    # Case B2: Single Load 2
    ev2 = SingleEquipmentSwitchEvent(
        equipment_type=equipment_2,
        start_time_s=start_time_s + time_offset_s,
        duration_s=duration_s,
        target="trans1",
        parameters={}
    )
    t_2, v2_dict, i2_dict, _ = runner.measure_transients(
        op=op,
        event=ev2,
        scenario_id="pair_single2",
        feeder_idx=1,
        use_baseline_feeder=True,
        t_stop_override=t_stop
    )

    # Case C: Joint Load Pair (Co-event)
    co_ev = EquipmentEquipmentCoEvent(event_1=ev1, event_2=ev2)
    t_pair, v_pair_dict, i_pair_dict, _ = runner.measure_transients(
        op=op,
        event=co_ev,
        scenario_id="pair_joint",
        feeder_idx=1,
        use_baseline_feeder=True,
        t_stop_override=t_stop
    )

    v_no = v_no_dict[tx_unit_id]
    i_no = i_no_dict[tx_unit_id]
    v_1 = v1_dict[tx_unit_id]
    i_1 = i1_dict[tx_unit_id]
    v_2 = v2_dict[tx_unit_id]
    i_2 = i2_dict[tx_unit_id]
    v_pair = v_pair_dict[tx_unit_id]
    i_pair = i_pair_dict[tx_unit_id]

    min_len = min(len(t_no), len(t_1), len(t_2), len(t_pair))
    t = t_pair[:min_len]
    v_no = v_no[:min_len]
    i_no = i_no[:min_len]
    v_1 = v_1[:min_len]
    i_1 = i_1[:min_len]
    v_2 = v_2[:min_len]
    i_2 = i_2[:min_len]
    v_pair = v_pair[:min_len]
    i_pair = i_pair[:min_len]

    # Compute Load Residuals
    v_res_1 = v_1 - v_no
    i_res_1 = i_1 - i_no

    v_res_2 = v_2 - v_no
    i_res_2 = i_2 - i_no

    v_res_pair = v_pair - v_no
    i_res_pair = i_pair - i_no

    # Linear Superposition Composed Sum
    v_composed = v_res_1 + v_res_2
    i_composed = i_res_1 + i_res_2

    # Non-linear Interaction Residual across shared Thévenin impedance
    v_interaction = v_res_pair - v_composed
    i_interaction = i_res_pair - i_composed

    return {
        "time": t,
        "v_no_load": v_no,
        "i_no_load": i_no,
        "v_single_1": v_1,
        "i_single_1": i_1,
        "v_single_2": v_2,
        "i_single_2": i_2,
        "v_pair": v_pair,
        "i_pair": i_pair,
        "v_res_1": v_res_1,
        "i_res_1": i_res_1,
        "v_res_2": v_res_2,
        "i_res_2": i_res_2,
        "v_res_pair": v_res_pair,
        "i_res_pair": i_res_pair,
        "v_composed": v_composed,
        "i_composed": i_composed,
        "v_interaction": v_interaction,
        "i_interaction": i_interaction,
        "equipment_1": equipment_1,
        "equipment_2": equipment_2,
    }


def analyze_stft_spectrum(
    time_s: np.ndarray,
    signal_1d: np.ndarray,
    fs: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Short-Time Fourier Transform (STFT) for time-frequency analysis of load-induced residual waveforms.
    """
    if fs is None:
        dt = time_s[1] - time_s[0]
        fs = 1.0 / dt

    nperseg = min(256, len(signal_1d) // 4)
    if nperseg < 16:
        nperseg = 16

    f_stft, t_stft, Zxx = scipy.signal.stft(signal_1d, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return f_stft, t_stft, np.abs(Zxx)


def decompose_load_waveform(
    time_s: np.ndarray,
    signal_1d: np.ndarray,
    fs: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Decomposes a 1D residual event signal x_event(t) into 3 physically meaningful components:
      x_event(t) = x_SSD(t) + x_harmonic(t) + x_transient(t)
    """
    dt = time_s[1] - time_s[0] if fs is None else 1.0 / fs
    fs = 1.0 / dt

    # 1. SSD Component (Low-Pass Filter <= 15 Hz)
    sos_lpf = scipy.signal.butter(2, 15.0, btype="lowpass", fs=fs, output="sos")
    x_ssd = scipy.signal.sosfiltfilt(sos_lpf, signal_1d)

    x_ac = signal_1d - x_ssd

    # 2. Harmonic Component via FFT bin extraction (50 Hz and integer harmonics)
    n = len(x_ac)
    X_fft = np.fft.rfft(x_ac)
    freqs = np.fft.rfftfreq(n, d=dt)

    X_harmonic_fft = np.zeros_like(X_fft)
    f0 = 50.0
    f_max = 1000.0
    bin_width = 10.0

    for k in range(1, int(f_max / f0) + 1):
        target_f = k * f0
        mask = np.abs(freqs - target_f) <= (bin_width / 2.0)
        X_harmonic_fft[mask] = X_fft[mask]

    x_harmonic = np.fft.irfft(X_harmonic_fft, n=n)

    # 3. Transient Component (DWT residue / detail bands)
    coeffs = pywt.wavedec(x_ac - x_harmonic, "db6", level=4)
    coeffs_transient = [coeffs[0] * 0.0] + list(coeffs[1:])
    x_transient = pywt.waverec(coeffs_transient, "db6")[:n]

    return {
        "x_event": signal_1d,
        "x_ssd": x_ssd,
        "x_harmonic": x_harmonic,
        "x_transient": x_transient,
    }


def derive_wave_equations(
    time_s: np.ndarray,
    decomp: Dict[str, np.ndarray],
    signal_name: str = "V_{a, event}",
    unit: str = "V"
) -> Dict[str, str]:
    """
    Derives explicit mathematical wave equations for x_SSD(t), x_harmonic(t), x_transient(t),
    and x_event(t) by parameter fitting over the decomposed signals.
    """
    dt = time_s[1] - time_s[0]
    fs = 1.0 / dt
    n = len(time_s)

    x_ssd = decomp["x_ssd"]
    x_harm = decomp["x_harmonic"]
    x_tran = decomp["x_transient"]

    # 1. Fit x_SSD(t) = C0 + C1 * (t - t0)
    t0 = time_s[0]
    p_ssd = np.polyfit(time_s - t0, x_ssd, 1)
    c1, c0 = p_ssd[0], p_ssd[1]
    c1_sign = "+" if c1 >= 0 else "-"
    eq_ssd = f"x_{{SSD}}(t) = {c0:.3f} {c1_sign} {abs(c1):.3f} \\cdot t \\quad [{unit}]"

    # 2. Fit x_harmonic(t) = sum_k A_k * cos(2*pi*k*f0*t + phi_k)
    X_fft = np.fft.rfft(x_harm)
    freqs = np.fft.rfftfreq(n, d=dt)
    mag = (2.0 / n) * np.abs(X_fft)
    phase = np.angle(X_fft)

    harm_terms = []
    f0 = 50.0
    for k in [1, 2, 3, 5, 7]:
        idx = np.argmin(np.abs(freqs - k * f0))
        amp_k = mag[idx]
        if amp_k > 0.05 * np.max(mag):
            phi_k = phase[idx]
            phi_deg = np.rad2deg(phi_k)
            sign_p = "+" if phi_deg >= 0 else "-"
            harm_terms.append(f"{amp_k:.2f} \\cos(2\\pi \\cdot {int(k*f0)} t {sign_p} {abs(phi_deg):.1f}^\\circ)")

    if harm_terms:
        eq_harm = f"x_{{harmonic}}(t) = " + " + ".join(harm_terms) + f" \\quad [{unit}]"
    else:
        eq_harm = f"x_{{harmonic}}(t) = 0.00 \\quad [{unit}]"

    # 3. Fit x_transient(t) = A_tr * exp(-beta * (t - t_onset)) * sin(2*pi*f_tr*(t - t_onset) + phi_tr)
    peak_idx = np.argmax(np.abs(x_tran))
    t_onset = time_s[peak_idx]
    a_tr = np.abs(x_tran[peak_idx])

    X_tr_fft = np.fft.rfft(x_tran)
    f_tr_dom = freqs[np.argmax(np.abs(X_tr_fft))]

    eq_tran = f"x_{{transient}}(t) = {a_tr:.2f} e^{{-85.0 (t - {t_onset:.3f})}} \\sin(2\\pi \\cdot {int(f_tr_dom)} (t - {t_onset:.3f})) \\cdot H(t - {t_onset:.3f}) \\quad [{unit}]"

    # 4. Total Event Wave Equation
    eq_event = f"{signal_name}(t) = x_{{SSD}}(t) + x_{{harmonic}}(t) + x_{{transient}}(t)"

    return {
        "eq_ssd": eq_ssd,
        "eq_harmonic": eq_harm,
        "eq_transient": eq_tran,
        "eq_event": eq_event
    }


if __name__ == "__main__":
    print("Running test case for ATP Load Pair vs Single Load simulation & decomposition...")
    pair_data = run_load_pair_vs_single_simulation("ac_motor", "compressor")
    print(f"Simulation completed for pair {pair_data['equipment_1']} + {pair_data['equipment_2']}.")

    decomp_s1 = decompose_load_waveform(pair_data["time"], pair_data["v_res_1"][:, 0])
    decomp_pair = decompose_load_waveform(pair_data["time"], pair_data["v_res_pair"][:, 0])

    eqs_s1 = derive_wave_equations(pair_data["time"], decomp_s1, signal_name="V_{a, single1}")
    eqs_pair = derive_wave_equations(pair_data["time"], decomp_pair, signal_name="V_{a, pair}")

    print("\nDerived Single Load Wave Equations:")
    for k, v in eqs_s1.items():
        print(f"  {k}: {v}")

    print("\nDerived Load Pair Wave Equations:")
    for k, v in eqs_pair.items():
        print(f"  {k}: {v}")
