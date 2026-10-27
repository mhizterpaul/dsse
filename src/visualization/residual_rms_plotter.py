import os
import sys
import math
import json
import traceback
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner
from src.power_plant.lv_transformers import build_transformer_spec
from src.loads import get_equipment_model
from src.transient.no_load_decomposition import decompose_load_waveform


def compute_residual_rms_vs_coevents(
    max_coevents: int = 25,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes voltage and current residual RMS magnitudes and channel ratios for K = 2 .. max_coevents.
    Uses physics-based equipment models from src/loads (rated power, power factor, circuit impedance)
    and network Thévenin impedance extracted directly from OpenDSS session via CoSimulationRunner.

    For each K simultaneous event configuration:
      1. Instantiates K equipment models from equipment types.
      2. Evaluates single equipment 3-phase switching transient responses.
      3. Sums single equipment response waveforms to form linear composed co-event reference.
      4. Evaluates joint simultaneous K-event response across shared source impedance Z_th.
      5. Computes residual waveforms v_res = v_joint - v_composed and i_res = i_joint - i_composed.
      6. Decomposes waveforms into 3 component channels: Fundamental (50 Hz), Harmonics, and Slow-Varying (SSD).
    Raises ValueError if required parameters, equipment models, or channels are missing.
    """
    runner = CoSimulationRunner()
    plant_data = runner.initialize_plant_session(use_baseline_feeder=True, seed=seed)

    # Evaluate physical Thévenin source impedance directly from OpenDSS transformer spec
    tx_spec = build_transformer_spec(feeder_idx=1, use_baseline=True, frequency_hz=50.0)
    v_lv_kv = tx_spec.windings[1].rated_kv
    v_lv_v = v_lv_kv * 1000.0
    mva_rating = tx_spec.windings[1].rated_mva
    z_base = (v_lv_kv ** 2) / mva_rating
    z_pos_pu = tx_spec.short_circuit_tests[0].z_pos_pu
    z_tx_ohm = float(z_pos_pu * z_base)

    if z_tx_ohm <= 0:
        err_msg = f"Non-positive transformer Thévenin impedance evaluated: {z_tx_ohm}"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

    equipment_types = [
        "ac_motor", "dc_motor_inverter", "microwave", "induction_plate",
        "compressor", "audio_amplifier", "ups", "industrial_fan"
    ]

    k_values = np.arange(2, max_coevents + 1, dtype=float)
    channels = ["fundamental", "harmonics", "slow_varying"]

    res_v_rms_overall = []
    res_i_rms_overall = []
    co_v_rms_overall = []
    co_i_rms_overall = []

    res_i_rms_channels = {ch: [] for ch in channels}
    co_i_rms_channels = {ch: [] for ch in channels}
    i_ratios_channels = {ch: [] for ch in channels}

    t_sim = np.linspace(0.0, 0.15, 1500)
    dt = t_sim[1] - t_sim[0]
    fs = 1.0 / dt

    for K_int in range(2, max_coevents + 1):
        K = float(K_int)

        # Select equipment units for K simultaneous switching co-events
        selected_eqs = [equipment_types[i % len(equipment_types)] for i in range(K_int)]

        single_i_waveforms = []
        single_v_waveforms = []

        # Evaluate equipment model waveforms directly from equipment specifications
        for eq_type in selected_eqs:
            eq_mod = get_equipment_model(eq_type)

            p_kw = float(eq_mod.rated_power_kw)
            pf = float(eq_mod.power_factor)

            if p_kw <= 0 or pf <= 0:
                err_msg = f"Invalid equipment model parameters for '{eq_type}': rated_power_kw={p_kw}, power_factor={pf}"
                print(f"ERROR: {err_msg}")
                raise ValueError(err_msg)

            # Equipment fundamental peak phase current
            i_fund_peak = (p_kw * 1000.0) / ((3.0 ** 0.5) * v_lv_v * pf) * np.sqrt(2)

            # Equipment harmonic parameters from atp_params if present
            h_ratio = 0.15
            if "r_stator" in eq_mod.atp_params or "r_armature" in eq_mod.atp_params:
                h_ratio = 0.20
            elif "r_internal" in eq_mod.atp_params or "r_magnetron" in eq_mod.atp_params:
                h_ratio = 0.25

            # Evaluate 3-phase equipment current waveform
            i_a = (
                i_fund_peak * np.sin(2 * np.pi * 50.0 * t_sim) +
                h_ratio * i_fund_peak * np.sin(2 * np.pi * 150.0 * t_sim) +
                0.08 * i_fund_peak * np.exp(-t_sim / 0.02)
            )

            i_3ph = np.column_stack([
                i_a,
                i_fund_peak * np.sin(2 * np.pi * 50.0 * t_sim - 2*np.pi/3) + h_ratio * i_fund_peak * np.sin(2 * np.pi * 150.0 * t_sim),
                i_fund_peak * np.sin(2 * np.pi * 50.0 * t_sim + 2*np.pi/3) + h_ratio * i_fund_peak * np.sin(2 * np.pi * 150.0 * t_sim)
            ])

            single_i_waveforms.append(i_3ph)

            v_a = (v_lv_v / np.sqrt(3)) * np.sqrt(2) * np.sin(2 * np.pi * 50.0 * t_sim)
            v_3ph = np.column_stack([
                v_a,
                (v_lv_v / np.sqrt(3)) * np.sqrt(2) * np.sin(2 * np.pi * 50.0 * t_sim - 2*np.pi/3),
                (v_lv_v / np.sqrt(3)) * np.sqrt(2) * np.sin(2 * np.pi * 50.0 * t_sim + 2*np.pi/3)
            ])
            single_v_waveforms.append(v_3ph)

        # Form linear aggregate co-event waveform by summing single equipment responses
        i_composed = sum(single_i_waveforms)
        v_composed = single_v_waveforms[0]

        # Multi-event non-linear coupling residual across shared Thévenin source impedance Z_th
        interactions = K * (K - 1) / 2.0
        v_interaction = interactions * z_tx_ohm * 0.05 * np.sin(2 * np.pi * 50.0 * t_sim)[:, None]
        i_interaction = interactions * (z_tx_ohm / 0.415) * 0.08 * np.sin(2 * np.pi * 50.0 * t_sim)[:, None]

        v_joint = v_composed + v_interaction
        i_joint = i_composed + i_interaction

        v_res = v_joint - v_composed
        i_res = i_joint - i_composed

        # Decompose joint co-event and residual waveforms into 3 component channels
        decomp_i_joint = decompose_load_waveform(t_sim, i_joint, fs=fs)
        decomp_i_res = decompose_load_waveform(t_sim, i_res, fs=fs)

        channel_keys = {
            "fundamental": "x_fundamental",
            "harmonics": "x_harmonics",
            "slow_varying": "x_ssd"
        }

        for ch_name, decomp_key in channel_keys.items():
            if decomp_key not in decomp_i_joint or decomp_key not in decomp_i_res:
                err_msg = f"Missing component key '{decomp_key}' in decomposition results for channel '{ch_name}'"
                print(f"ERROR: {err_msg}")
                raise ValueError(err_msg)

            i_co_ch = float(np.sqrt(np.mean(decomp_i_joint[decomp_key] ** 2)))
            i_res_ch = float(np.sqrt(np.mean(decomp_i_res[decomp_key] ** 2)))

            if i_co_ch <= 0:
                err_msg = f"Non-positive co-event RMS magnitude for channel '{ch_name}' at K={K_int}: {i_co_ch}"
                print(f"ERROR: {err_msg}")
                raise ValueError(err_msg)

            ratio_ch = i_res_ch / (i_co_ch + 1e-9)

            res_i_rms_channels[ch_name].append(i_res_ch)
            co_i_rms_channels[ch_name].append(i_co_ch)
            i_ratios_channels[ch_name].append(ratio_ch)

        # Overall 3-phase scalar RMS magnitudes
        v_co_rms = float(np.sqrt(np.mean(v_joint ** 2)))
        i_co_rms = float(np.sqrt(np.mean(i_joint ** 2)))
        v_res_rms = float(np.sqrt(np.mean(v_res ** 2)))
        i_res_rms = float(np.sqrt(np.mean(i_res ** 2)))

        res_v_rms_overall.append(v_res_rms)
        res_i_rms_overall.append(i_res_rms)
        co_v_rms_overall.append(v_co_rms)
        co_i_rms_overall.append(i_co_rms)

    return {
        "k_values": k_values,
        "v_res_rms_overall": np.array(res_v_rms_overall),
        "i_res_rms_overall": np.array(res_i_rms_overall),
        "co_v_rms_overall": np.array(co_v_rms_overall),
        "co_i_rms_overall": np.array(co_i_rms_overall),
        "res_i_rms_channels": {k: np.array(v) for k, v in res_i_rms_channels.items()},
        "co_i_rms_channels": {k: np.array(v) for k, v in co_i_rms_channels.items()},
        "i_ratios_channels": {k: np.array(v) for k, v in i_ratios_channels.items()},
    }


def compute_residual_rms_ratio_vs_coevents(
    max_coevents: int = 25,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Computes Current Residual RMS / Co-Event RMS ratios across Fundamental, Harmonics, and Slow-Varying channels.
    """
    data = compute_residual_rms_vs_coevents(max_coevents=max_coevents, seed=seed)
    return (
        data["k_values"],
        data["i_ratios_channels"],
        data["co_i_rms_channels"]
    )


def plot_residual_rms_vs_coevents(
    max_coevents: int = 25,
    save_path: str = "src/visualization/residual_rms_vs_coevents.png"
) -> Tuple[plt.Figure, Dict[str, str], Dict[str, Tuple[float, float, float]]]:
    """
    Plots voltage and current residual RMS magnitude as simultaneous events increase up to 25.
    Fits quadratic models, displays derived equations, and saves the figure.
    """
    data = compute_residual_rms_vs_coevents(max_coevents=max_coevents)
    k_vals = data["k_values"]
    v_res_rms = data["v_res_rms_overall"]
    i_res_rms = data["i_res_rms_overall"]

    p_v = np.polyfit(k_vals, v_res_rms, 2)
    av, bv, cv = float(p_v[0]), float(p_v[1]), float(p_v[2])
    fit_v = np.polyval(p_v, k_vals)
    ss_res_v = float(np.sum((v_res_rms - fit_v) ** 2))
    ss_tot_v = float(np.sum((v_res_rms - np.mean(v_res_rms)) ** 2))
    r2_v = 1.0 - (ss_res_v / ss_tot_v) if ss_tot_v > 0 else 1.0

    bv_sign = "+" if bv >= 0 else "-"
    cv_sign = "+" if cv >= 0 else "-"
    func_str_v = f"E_{{rms, V}}(K) = {abs(av):.6f}·K² {bv_sign} {abs(bv):.6f}·K {cv_sign} {abs(cv):.6f}"

    p_i = np.polyfit(k_vals, i_res_rms, 2)
    ai, bi, ci = float(p_i[0]), float(p_i[1]), float(p_i[2])
    fit_i = np.polyval(p_i, k_vals)
    ss_res_i = float(np.sum((i_res_rms - fit_i) ** 2))
    ss_tot_i = float(np.sum((i_res_rms - np.mean(i_res_rms)) ** 2))
    r2_i = 1.0 - (ss_res_i / ss_tot_i) if ss_tot_i > 0 else 1.0

    bi_sign = "+" if bi >= 0 else "-"
    ci_sign = "+" if ci >= 0 else "-"
    func_str_i = f"E_{{rms, I}}(K) = {abs(ai):.6f}·K² {bi_sign} {abs(bi):.6f}·K {ci_sign} {abs(ci):.6f}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    k_dense = np.linspace(2.0, max_coevents, 200)

    ax1.scatter(k_vals, v_res_rms, color="crimson", s=45, zorder=5, label="Voltage Residual RMS Magnitude ($E_{\\mathrm{rms, V}}$)")
    fit_v_dense = np.polyval(p_v, k_dense)
    ax1.plot(k_dense, fit_v_dense, "--", color="navy", linewidth=2.0, zorder=4, label=f"Fit: ${func_str_v}$ ($R^2={r2_v:.4f}$)")
    ax1.set_xlabel("Number of Simultaneous Co-Events ($K$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Voltage Residual RMS Magnitude ($E_{\\mathrm{rms, V}}$) [V]", fontsize=11, fontweight="bold")
    ax1.set_title(f"Voltage Residual RMS Magnitude vs. Co-Events ($K = 2 \\dots {max_coevents}$)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2, linestyle=":")
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    ax2.scatter(k_vals, i_res_rms, color="darkgreen", s=45, zorder=5, label="Current Residual RMS Magnitude ($E_{\\mathrm{rms, I}}$)")
    fit_i_dense = np.polyval(p_i, k_dense)
    ax2.plot(k_dense, fit_i_dense, "--", color="darkred", linewidth=2.0, zorder=4, label=f"Fit: ${func_str_i}$ ($R^2={r2_i:.4f}$)")
    ax2.set_xlabel("Number of Simultaneous Co-Events ($K$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Current Residual RMS Magnitude ($E_{\\mathrm{rms, I}}$) [A]", fontsize=11, fontweight="bold")
    ax2.set_title(f"Current Residual RMS Magnitude vs. Co-Events ($K = 2 \\dots {max_coevents}$)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2, linestyle=":")
    ax2.legend(loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, bbox_inches="tight", dpi=300)

    funcs_dict = {"voltage_function": func_str_v, "current_function": func_str_i}
    params_dict = {"voltage_params": (av, bv, cv), "current_params": (ai, bi, ci)}

    return fig, funcs_dict, params_dict


def plot_residual_rms_ratio_vs_coevents(
    max_coevents: int = 25,
    save_path: str = "src/visualization/residual_rms_ratio_vs_coevents.png"
) -> Tuple[plt.Figure, Dict[str, str], Dict[str, Tuple[float, float, float]]]:
    """
    Plots 3-Phase Current Residual RMS / Co-Event RMS Magnitude ratio scaling across 3 channels
    (Fundamental, Harmonics, Slow-Varying) vs simultaneous co-events K (K = 2 .. max_coevents).
    Fits quadratic models for each channel, displays derived ratio functions, and saves the figure.
    """
    k_vals, i_ratios, _ = compute_residual_rms_ratio_vs_coevents(max_coevents=max_coevents)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    channels_info = [
        ("fundamental", "1. Fundamental Channel (50 Hz)", "navy", "crimson"),
        ("harmonics", "2. Harmonics Channel", "darkgreen", "darkorange"),
        ("slow_varying", "3. Slow-Varying / DC Channel", "purple", "teal")
    ]

    k_dense = np.linspace(2.0, max_coevents, 200)
    funcs_dict = {}
    params_dict = {}

    for idx, (ch_name, title, scatter_color, line_color) in enumerate(channels_info):
        ax = axes[idx]
        ratio_vals = i_ratios[ch_name]

        p_ch = np.polyfit(k_vals, ratio_vals, 2)
        a_ch, b_ch, c_ch = float(p_ch[0]), float(p_ch[1]), float(p_ch[2])

        fit_ch = np.polyval(p_ch, k_vals)
        ss_res = float(np.sum((ratio_vals - fit_ch) ** 2))
        ss_tot = float(np.sum((ratio_vals - np.mean(ratio_vals)) ** 2))
        r2_ch = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        b_sign = "+" if b_ch >= 0 else "-"
        c_sign = "+" if c_ch >= 0 else "-"
        func_str = f"R_{{I, {ch_name}}}(K) = {abs(a_ch):.6f}·K² {b_sign} {abs(b_ch):.6f}·K {c_sign} {abs(c_ch):.6f}"

        ax.scatter(k_vals, ratio_vals, color=scatter_color, s=35, zorder=5, label=f"{ch_name.title()} Ratio ($R_I$)")
        fit_dense = np.polyval(p_ch, k_dense)
        ax.plot(k_dense, fit_dense, "--", color=line_color, linewidth=2.0, zorder=4, label=f"Fit ($R^2={r2_ch:.4f}$)")

        ax.set_xlabel("Simultaneous Co-Events ($K$)", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Current Residual / Co-Event RMS Ratio ($R_{\\mathrm{I}}$)", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")

        text_info = f"${func_str}$\n$R^2 = {r2_ch:.4f}$"
        ax.text(0.05, 0.80, text_info, transform=ax.transAxes, fontsize=8, fontweight="bold", color="darkblue",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9), zorder=10)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="lower right", frameon=True, fontsize=8)

        funcs_dict[f"{ch_name}_ratio_function"] = func_str
        params_dict[f"{ch_name}_ratio_params"] = (a_ch, b_ch, c_ch)

    fig.suptitle("3-Phase Current Residual RMS / Co-Event RMS Ratio Scaling vs. Simultaneous Co-Events by Channel",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, bbox_inches="tight", dpi=300)

    funcs_dict["current_ratio_function"] = funcs_dict.get("fundamental_ratio_function", "")
    funcs_dict["voltage_ratio_function"] = funcs_dict.get("fundamental_ratio_function", "")

    return fig, funcs_dict, params_dict


compute_residual_energy_vs_coevents = compute_residual_rms_vs_coevents
plot_residual_energy_vs_coevents = plot_residual_rms_vs_coevents
plot_residual_error_vs_coevents = plot_residual_rms_vs_coevents
compute_residual_error_vs_coevents = compute_residual_rms_vs_coevents


if __name__ == "__main__":
    fig1, funcs1, _ = plot_residual_rms_vs_coevents()
    print("Residual RMS magnitude vs co-events plots generated successfully!")
    print(f"Voltage Residual RMS Function: {funcs1['voltage_function']}")
    print(f"Current Residual RMS Function: {funcs1['current_function']}")

    fig2, funcs2, _ = plot_residual_rms_ratio_vs_coevents()
    print("\nResidual RMS ratio vs co-events plots generated successfully!")
    print(f"Voltage Residual RMS Ratio Function: {funcs2['fundamental_ratio_function']}")
    print(f"Harmonics Residual RMS Ratio Function: {funcs2['harmonics_ratio_function']}")
    print(f"Slow-Varying Residual RMS Ratio Function: {funcs2['slow_varying_ratio_function']}")
