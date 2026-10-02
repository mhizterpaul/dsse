import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
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


def run_clean_experiment_simulations(
    equipment_1: str = "ac_motor",
    equipment_2: str = "compressor",
    start_time_s: float = 0.02,
    duration_s: float = 0.10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the clean experiment simulation suite ONCE under identical ATP circuit conditions:
      - Case A:  No-Load baseline (secondary open circuit)
      - Case B1: Single Equipment 1 (e.g., AC Motor)
      - Case B2: Single Equipment 2 (e.g., Compressor)
      - Case C:  Joint Equipment Pair (Equipment 1 + Equipment 2)

    Computes load residuals by subtracting the single Case A no-load baseline waveform:
      - x_res_1 = x_single_1 - x_no_load
      - x_res_2 = x_single_2 - x_no_load
      - x_res_pair = x_pair - x_no_load
      - x_composed = x_res_1 + x_res_2
      - x_interaction = x_res_pair - x_composed
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

    # 1. Case A: Single No-Load Baseline Run
    no_load_ev = NoLoadEvent(start_time_s=start_time_s, duration_s=duration_s)
    t_no, v_no_dict, i_no_dict, _ = runner.measure_transients(
        op=op,
        event=no_load_ev,
        scenario_id="clean_case_a_noload",
        feeder_idx=1,
        use_baseline_feeder=True,
        t_stop_override=t_stop
    )

    # 2. Case B1: Single Equipment 1
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
        scenario_id="clean_case_b1_single1",
        feeder_idx=1,
        use_baseline_feeder=True,
        t_stop_override=t_stop
    )

    # 3. Case B2: Single Equipment 2
    ev2 = SingleEquipmentSwitchEvent(
        equipment_type=equipment_2,
        start_time_s=start_time_s,
        duration_s=duration_s,
        target="trans1",
        parameters={}
    )
    t_2, v2_dict, i2_dict, _ = runner.measure_transients(
        op=op,
        event=ev2,
        scenario_id="clean_case_b2_single2",
        feeder_idx=1,
        use_baseline_feeder=True,
        t_stop_override=t_stop
    )

    # 4. Case C: Joint Equipment Pair (Co-Event)
    co_ev = EquipmentEquipmentCoEvent(event_1=ev1, event_2=ev2)
    t_pair, v_pair_dict, i_pair_dict, _ = runner.measure_transients(
        op=op,
        event=co_ev,
        scenario_id="clean_case_c_joint_pair",
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

    # Subtract the single Case A no-load baseline
    v_res_1 = v_1 - v_no
    i_res_1 = i_1 - i_no

    v_res_2 = v_2 - v_no
    i_res_2 = i_2 - i_no

    v_res_pair = v_pair - v_no
    i_res_pair = i_pair - i_no

    # Linear Superposition Composed Sum
    v_composed = v_res_1 + v_res_2
    i_composed = i_res_1 + i_res_2

    # Non-linear Interaction Residual across shared Thévenin source impedance
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


run_no_load_vs_loaded_simulation = run_clean_experiment_simulations
run_load_pair_vs_single_simulation = run_clean_experiment_simulations


def decompose_load_waveform(
    time_s: np.ndarray,
    signal: np.ndarray,
    fs: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Decomposes a 1D or 2D (N, 3) event signal x_event(t) into 5 physically meaningful components:
      1. Fundamental Component (50 Hz fundamental frequency)
      2. Harmonic Bands (100 Hz, 150 Hz, 200 Hz... integer harmonics)
      3. Localized Transient (Short-duration pulse / switching onset impulse)
      4. Oscillatory Transient (High-frequency ringing / damped oscillation)
      5. Slowly Varying Component (SSD / DC trend <= 15 Hz)
    """
    signal_arr = np.asarray(signal)
    if signal_arr.ndim == 2:
        res = {
            "x_event": signal_arr,
            "x_fundamental": np.zeros_like(signal_arr),
            "x_harmonics": np.zeros_like(signal_arr),
            "x_localized_transient": np.zeros_like(signal_arr),
            "x_oscillatory_transient": np.zeros_like(signal_arr),
            "x_ssd": np.zeros_like(signal_arr),
        }
        for ch in range(signal_arr.shape[1]):
            d_ch = decompose_load_waveform(time_s, signal_arr[:, ch], fs=fs)
            for k in res:
                if k != "x_event":
                    res[k][:, ch] = d_ch[k]
        return res

    dt = time_s[1] - time_s[0] if fs is None else 1.0 / fs
    fs = 1.0 / dt
    n = len(signal_arr)

    # 1. Slowly Varying Component (SSD, LPF <= 15 Hz)
    sos_lpf = scipy.signal.butter(2, 15.0, btype="lowpass", fs=fs, output="sos")
    x_ssd = scipy.signal.sosfiltfilt(sos_lpf, signal_arr)

    x_ac = signal_arr - x_ssd

    # 2. Fundamental Component (50 Hz band) & Harmonic Bands (100, 150, 200... Hz)
    X_fft = np.fft.rfft(x_ac)
    freqs = np.fft.rfftfreq(n, d=dt)

    X_fund_fft = np.zeros_like(X_fft)
    mask_fund = np.abs(freqs - 50.0) <= 5.0
    X_fund_fft[mask_fund] = X_fft[mask_fund]
    x_fundamental = np.fft.irfft(X_fund_fft, n=n)

    X_harm_fft = np.zeros_like(X_fft)
    f_max = 1000.0
    for k in range(2, int(f_max / 50.0) + 1):
        mask_h = np.abs(freqs - k * 50.0) <= 5.0
        X_harm_fft[mask_h] = X_fft[mask_h]
    x_harmonics = np.fft.irfft(X_harm_fft, n=n)

    # 3. Transient Residual = x_ac - x_fundamental - x_harmonics
    x_transient_total = x_ac - x_fundamental - x_harmonics

    coeffs = pywt.wavedec(x_transient_total, "db6", level=4)
    coeffs_loc = [coeffs[0] * 0.0] + [coeffs[1] * 0.0, coeffs[2] * 0.0] + [coeffs[3], coeffs[4]]
    x_localized = pywt.waverec(coeffs_loc, "db6")[:n]

    x_oscillatory = x_transient_total - x_localized

    return {
        "x_event": signal_arr,
        "x_fundamental": x_fundamental,
        "x_harmonics": x_harmonics,
        "x_localized_transient": x_localized,
        "x_oscillatory_transient": x_oscillatory,
        "x_ssd": x_ssd,
    }


def derive_wave_equations(
    time_s: np.ndarray,
    decomp: Dict[str, np.ndarray],
    signal_name: str = "V",
    unit: str = "V"
) -> Dict[str, str]:
    """
    Derives explicit mathematical wave equations with fitted numerical values for all 5 decomposed components
    and full expanded total signal expressions. Supports both 1D arrays and 3-phase (N, 3) arrays.
    """
    x_event = decomp["x_event"]
    if x_event.ndim == 2 and x_event.shape[1] == 3:
        ph_names = ["a", "b", "c"]
        eqs = {}
        for p_idx, p_ch in enumerate(ph_names):
            d_p = {
                "x_event": decomp["x_event"][:, p_idx],
                "x_fundamental": decomp["x_fundamental"][:, p_idx],
                "x_harmonics": decomp["x_harmonics"][:, p_idx],
                "x_localized_transient": decomp["x_localized_transient"][:, p_idx],
                "x_oscillatory_transient": decomp["x_oscillatory_transient"][:, p_idx],
                "x_ssd": decomp["x_ssd"][:, p_idx],
            }
            eq_p = derive_wave_equations(time_s, d_p, signal_name=f"{signal_name}_{{{p_ch}}}", unit=unit)
            for k, v in eq_p.items():
                eqs[f"{k}_phase_{p_ch}"] = v
        return eqs

    dt = time_s[1] - time_s[0]
    fs = 1.0 / dt
    n = len(time_s)

    x_fund = decomp["x_fundamental"]
    x_harm = decomp["x_harmonics"]
    x_loc = decomp["x_localized_transient"]
    x_osc = decomp["x_oscillatory_transient"]
    x_ssd = decomp["x_ssd"]

    # 1. Fundamental Component: A_1 * cos(2*pi*50*t + phi_1)
    X_f_fft = np.fft.rfft(x_fund)
    freqs = np.fft.rfftfreq(n, d=dt)
    idx_50 = np.argmin(np.abs(freqs - 50.0))
    amp_f1 = (2.0 / n) * np.abs(X_f_fft[idx_50])
    phi_f1_deg = np.rad2deg(np.angle(X_f_fft[idx_50]))
    sign_f1 = "+" if phi_f1_deg >= 0 else "-"
    term_fund = f"{amp_f1:.2f} \\cos(2\\pi \\cdot 50 t {sign_f1} {abs(phi_f1_deg):.1f}^\\circ)"
    eq_fund = f"x_{{fundamental}}(t) = {term_fund} \\quad [{unit}]"

    # 2. Harmonic Bands: sum_k A_k * cos(2*pi*k*50*t + phi_k)
    X_h_fft = np.fft.rfft(x_harm)
    mag_h = (2.0 / n) * np.abs(X_h_fft)
    phase_h = np.angle(X_h_fft)
    harm_terms = []
    for k in [2, 3, 5, 7]:
        idx_k = np.argmin(np.abs(freqs - k * 50.0))
        amp_k = mag_h[idx_k]
        if amp_k > 0.02 * np.max(mag_h):
            phi_k_deg = np.rad2deg(phase_h[idx_k])
            sign_k = "+" if phi_k_deg >= 0 else "-"
            harm_terms.append(f"{amp_k:.2f} \\cos(2\\pi \\cdot {int(k*50)} t {sign_k} {abs(phi_k_deg):.1f}^\\circ)")

    term_harm = " + ".join(harm_terms) if harm_terms else "0.00"
    eq_harm = f"x_{{harmonics}}(t) = {term_harm} \\quad [{unit}]"

    # 3. Localized Transient Impulse
    peak_loc_idx = np.argmax(np.abs(x_loc))
    t_loc = time_s[peak_loc_idx]
    amp_loc = np.abs(x_loc[peak_loc_idx])
    term_loc = f"{amp_loc:.2f} \\cdot \\mathrm{{rect}}\\!\\left(\\frac{{t - {t_loc:.3f}}}{{0.005}}\\right)"
    eq_loc = f"x_{{localized}}(t) = {term_loc} \\quad [{unit}]"

    # 4. Oscillatory Transient Ringing
    peak_osc_idx = np.argmax(np.abs(x_osc))
    t_osc = time_s[peak_osc_idx]
    amp_osc = np.abs(x_osc[peak_osc_idx])
    X_o_fft = np.fft.rfft(x_osc)
    f_osc_dom = freqs[np.argmax(np.abs(X_o_fft))]
    term_osc = f"{amp_osc:.2f} e^{{-120.0 (t - {t_osc:.3f})}} \\sin(2\\pi \\cdot {int(f_osc_dom)} (t - {t_osc:.3f})) \\cdot H(t - {t_osc:.3f})"
    eq_osc = f"x_{{oscillatory}}(t) = {term_osc} \\quad [{unit}]"

    # 5. Slowly Varying Component (SSD)
    t0 = time_s[0]
    p_ssd = np.polyfit(time_s - t0, x_ssd, 1)
    c1, c0 = p_ssd[0], p_ssd[1]
    c1_sign = "+" if c1 >= 0 else "-"
    term_ssd = f"{c0:.3f} {c1_sign} {abs(c1):.3f} \\cdot t"
    eq_ssd = f"x_{{SSD}}(t) = {term_ssd} \\quad [{unit}]"

    # Expanded Total Signal Expression with Fitted Numerical Values
    eq_event = f"{signal_name}(t) = \\left[ {term_fund} \\right] + \\left[ {term_harm} \\right] + \\left[ {term_loc} \\right] + \\left[ {term_osc} \\right] + \\left[ {term_ssd} \\right] \\quad [{unit}]"

    return {
        "eq_fundamental": eq_fund,
        "eq_harmonics": eq_harm,
        "eq_localized": eq_loc,
        "eq_oscillatory": eq_osc,
        "eq_ssd": eq_ssd,
        "eq_event": eq_event,
    }


def analyze_stft_spectrum(
    time_s: np.ndarray,
    signal_1d: np.ndarray,
    fs: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fs is None:
        dt = time_s[1] - time_s[0]
        fs = 1.0 / dt

    nperseg = min(256, len(signal_1d) // 4)
    if nperseg < 16:
        nperseg = 16

    f_stft, t_stft, Zxx = scipy.signal.stft(signal_1d, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return f_stft, t_stft, np.abs(Zxx)


if __name__ == "__main__":
    print("Testing 3-phase multi-channel 5-component expanded wave equations...")
    sim_data = run_clean_experiment_simulations("ac_motor", "compressor")
    decomp_v3 = decompose_load_waveform(sim_data["time"], sim_data["v_pair"])
    eqs_v3 = derive_wave_equations(sim_data["time"], decomp_v3, signal_name="V_{pair}")
    print("Derived Expanded 3-Phase Voltage Equations:")
    for k, v in eqs_v3.items():
        print(f"  {k}: {v}")
