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
from src.transient.events import SingleEquipmentSwitchEvent, EquipmentEquipmentCoEvent, MultiEquipmentCoEvent, NoLoadEvent
from src.transient.no_load_decomposition import decompose_load_waveform


def compute_residual_rms_vs_coevents(
    max_coevents: int = 25,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes voltage and current residual RMS magnitudes and channel ratios for K = 2 .. max_coevents
    directly using CoSimulationRunner and ATP-EMTP / OpenDSS pre-event Thévenin state estimation.

    For each K simultaneous event configuration:
      1. Subtracts no-load baseline to obtain event-induced voltage/current responses v_delta and i_delta.
      2. Generates single equipment switching responses and joint simultaneous K-event responses.
      3. Computes linear composed co-event reference by summing single equipment event-induced responses.
      4. Computes residual waveforms v_res = v_composed - v_joint_delta and i_res = i_composed - i_joint_delta.
      5. Decomposes baseline-subtracted joint and composed responses into 3 component channels:
         Fundamental (50 Hz), Harmonics, and Slow-Varying (SSD).
    Raises ValueError if required parameters, equipment models, or channels are missing.
    """
    runner = CoSimulationRunner()
    plant_data = runner.initialize_plant_session(use_baseline_feeder=True, seed=seed)

    # Disable steady-state loads upfront so single responses and joint responses share identical base operating point
    runner.dss.run_command("disable Load.*")
    from src.power_plant.plant import solve_operating_point
    op = solve_operating_point(runner.dss)

    tx_unit_id = "trans1_lv_boundary"

    # Measure no-switch baseline simulation under the same operating point with explicit 0.17s t_stop_override
    no_load_ev = NoLoadEvent(start_time_s=0.02, duration_s=0.10)
    try:
        t_s_base, v_base_dict, i_base_dict, _ = runner.measure_transients(
            op=op, event=no_load_ev, scenario_id="no_load_baseline", feeder_idx=1, use_baseline_feeder=True, t_stop_override=0.17
        )
        v_base = v_base_dict[tx_unit_id]
        i_base = i_base_dict[tx_unit_id]
    except Exception as e:
        err_msg = f"Failed to measure no-load baseline response: {e}"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg) from e

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

    res_v_rms_channels = {ch: [] for ch in channels}
    res_i_rms_channels = {ch: [] for ch in channels}
    co_v_rms_channels = {ch: [] for ch in channels}
    co_i_rms_channels = {ch: [] for ch in channels}
    i_ratios_channels = {ch: [] for ch in channels}

    # Pre-evaluate single equipment switching responses (event-induced delta) for all load models
    single_responses = {}
    for idx, eq_type in enumerate(equipment_types):
        ev = SingleEquipmentSwitchEvent(
            equipment_type=eq_type,
            start_time_s=0.02,
            duration_s=0.10,
            target=f"trans1_eq_{idx}",
            parameters={}
        )
        try:
            t_s, v_dict, i_dict, _ = runner.measure_transients(
                op=op, event=ev, scenario_id=f"single_{eq_type}", feeder_idx=1, use_baseline_feeder=True, t_stop_override=0.17
            )
            v_sig = v_dict[tx_unit_id]
            i_sig = i_dict[tx_unit_id]
            v_delta = v_sig - v_base
            i_delta = i_sig - i_base
            single_responses[eq_type] = (t_s, v_delta, i_delta)
        except Exception as e:
            err_msg = f"Failed to measure transient response for equipment '{eq_type}': {e}"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg) from e

    if len(single_responses) < len(equipment_types):
        err_msg = f"Incomplete single responses: expected {len(equipment_types)}, got {len(single_responses)}"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

    previous_i_joint = None

    channel_keys = {
        "fundamental": "x_fundamental",
        "harmonics": "x_harmonics",
        "slow_varying": "x_ssd"
    }

    # Evaluate for each K simultaneous co-events using CoSimulationRunner transients
    for K_int in range(2, max_coevents + 1):
        K = float(K_int)
        selected_eqs = [equipment_types[i % len(equipment_types)] for i in range(K_int)]

        t_sim = single_responses[selected_eqs[0]][0]

        i_composed = sum(single_responses[eq][2] for eq in selected_eqs)
        v_composed = sum(single_responses[eq][1] for eq in selected_eqs)

        # Construct multi-equipment co-event with keyword arguments for ALL K equipment switched simultaneously
        single_events = [
            SingleEquipmentSwitchEvent(
                equipment_type=eq_type,
                start_time_s=0.02,
                duration_s=0.10,
                target=f"trans1_eq_{idx}",
                parameters={},
            )
            for idx, eq_type in enumerate(selected_eqs)
        ]
        co_ev = MultiEquipmentCoEvent(events=single_events)

        t_s, v_dict, i_dict, _ = runner.measure_transients(
            op=op, event=co_ev, scenario_id=f"coevent_K{K_int}", feeder_idx=1, use_baseline_feeder=True, t_stop_override=0.17
        )

        if len(t_s) != len(t_sim) or not np.allclose(t_s, t_sim, rtol=0.0, atol=1e-9):
            raise ValueError(f"Time-vector mismatch for K={K_int}: len(t_s)={len(t_s)}, len(t_sim)={len(t_sim)}")

        fs_joint = 1.0 / (t_s[1] - t_s[0])

        v_joint = v_dict[tx_unit_id]
        i_joint = i_dict[tx_unit_id]

        v_joint_delta = v_joint - v_base
        i_joint_delta = i_joint - i_base

        v_res = v_composed - v_joint_delta
        i_res = i_composed - i_joint_delta

        joint_curr_rms = float(np.sqrt(np.mean(i_joint ** 2)))
        joint_delta_rms = float(np.sqrt(np.mean(i_joint_delta ** 2)))
        composed_rms = float(np.sqrt(np.mean(i_composed ** 2)))
        residual_rms = float(np.sqrt(np.mean(i_res ** 2)))

        if previous_i_joint is not None:
            relative_change = float(
                np.linalg.norm(i_joint - previous_i_joint)
                / (np.linalg.norm(previous_i_joint) + 1e-12)
            )
        previous_i_joint = i_joint.copy()

        # Decompose baseline-subtracted joint and composed responses using consistent method
        decomp_v_joint = decompose_load_waveform(t_s, v_joint_delta, fs=fs_joint)
        decomp_v_composed = decompose_load_waveform(t_s, v_composed, fs=fs_joint)

        decomp_i_joint = decompose_load_waveform(t_s, i_joint_delta, fs=fs_joint)
        decomp_i_composed = decompose_load_waveform(t_s, i_composed, fs=fs_joint)

        v_co_rms = float(np.sqrt(np.mean(v_joint_delta ** 2)))
        i_co_rms = float(np.sqrt(np.mean(i_joint_delta ** 2)))
        v_res_rms = float(np.sqrt(np.mean(v_res ** 2)))
        i_res_rms = float(np.sqrt(np.mean(i_res ** 2)))

        for ch_name, decomp_key in channel_keys.items():
            if decomp_key not in decomp_i_joint or decomp_key not in decomp_v_joint:
                err_msg = f"Missing component key '{decomp_key}' in decomposition results"
                print(f"ERROR: {err_msg}")
                raise ValueError(err_msg)

            v_joint_ch = decomp_v_joint[decomp_key]
            v_composed_ch = decomp_v_composed[decomp_key]
            v_res_ch_wave = v_composed_ch - v_joint_ch
            v_comp_ch = float(np.sqrt(np.mean(v_composed_ch ** 2)))
            v_res_ch = float(np.sqrt(np.mean(v_res_ch_wave ** 2)))

            i_joint_ch = decomp_i_joint[decomp_key]
            i_composed_ch = decomp_i_composed[decomp_key]

            if ch_name == "slow_varying":
                # For DC / slow-varying channel: residual is joint - composed
                i_res_ch_wave = i_joint_ch - i_composed_ch
                i_joint_rms = float(np.sqrt(np.mean(i_joint_ch ** 2)))
                i_res_ch = float(np.sqrt(np.mean(i_res_ch_wave ** 2)))
                i_comp_ch = float(np.sqrt(np.mean(i_composed_ch ** 2)))
                ratio_ch = i_res_ch / max(i_joint_rms, 1e-9)
            else:
                i_res_ch_wave = i_composed_ch - i_joint_ch
                i_comp_ch = float(np.sqrt(np.mean(i_composed_ch ** 2)))
                i_res_ch = float(np.sqrt(np.mean(i_res_ch_wave ** 2)))
                ratio_ch = i_res_ch / max(i_comp_ch, 1e-9)

            if ratio_ch >= 1.0:
                print(f"WARNING: Channel '{ch_name}' residual RMS ratio {ratio_ch:.4f} at K={K_int} is greater than or equal to 1.0.")

            res_v_rms_channels[ch_name].append(v_res_ch)
            co_v_rms_channels[ch_name].append(v_comp_ch)
            res_i_rms_channels[ch_name].append(i_res_ch)
            co_i_rms_channels[ch_name].append(i_comp_ch)
            i_ratios_channels[ch_name].append(ratio_ch)

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
        "res_v_rms_channels": {k: np.array(v) for k, v in res_v_rms_channels.items()},
        "co_v_rms_channels": {k: np.array(v) for k, v in co_v_rms_channels.items()},
        "res_i_rms_channels": {k: np.array(v) for k, v in res_i_rms_channels.items()},
        "co_i_rms_channels": {k: np.array(v) for k, v in co_i_rms_channels.items()},
        "i_ratios_channels": {k: np.array(v) for k, v in i_ratios_channels.items()},
    }


def plot_residual_rms_vs_coevents(
    max_coevents: int = 25,
    save_path: str = "src/visualization/residual_rms_vs_coevents.png"
) -> Tuple[plt.Figure, Dict[str, str], Dict[str, Tuple[float, float, float]]]:
    """
    Plots fundamental channel voltage and current residual RMS magnitude as simultaneous events increase up to 25.
    Fits polynomial models, displays derived equations, and saves the figure.
    """
    data = compute_residual_rms_vs_coevents(max_coevents=max_coevents)
    k_vals = data["k_values"]
    v_res_rms = data["res_v_rms_channels"]["fundamental"]
    i_res_rms = data["res_i_rms_channels"]["fundamental"]

    if v_res_rms is None or len(v_res_rms) == 0 or not np.all(np.isfinite(v_res_rms)):
        err_msg = f"Cannot fit curve: Voltage residual RMS values are missing or invalid: {v_res_rms}"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

    if len(k_vals) != len(v_res_rms):
        err_msg = f"Cannot fit curve: Mismatch between K values ({len(k_vals)}) and voltage residual RMS ({len(v_res_rms)})"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

    # Fit line of best fit for voltage residual RMS magnitude: E_{rms, V}(K) = a * K + b
    p_v = np.polyfit(k_vals, v_res_rms, 1)
    av, bv = float(p_v[0]), float(p_v[1])
    fit_v = np.polyval(p_v, k_vals)
    ss_res_v = float(np.sum((v_res_rms - fit_v) ** 2))
    ss_tot_v = float(np.sum((v_res_rms - np.mean(v_res_rms)) ** 2))
    r2_v = 1.0 - (ss_res_v / ss_tot_v) if ss_tot_v > 0 else 1.0

    bv_sign = "+" if bv >= 0 else "-"
    func_str_v = f"E_{{rms, V}}(K) = {abs(av):.6f}·K {bv_sign} {abs(bv):.6f}"

    if i_res_rms is None or len(i_res_rms) == 0 or not np.all(np.isfinite(i_res_rms)):
        err_msg = f"Cannot fit curve: Current residual RMS values are missing or invalid: {i_res_rms}"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

    if len(k_vals) != len(i_res_rms):
        err_msg = f"Cannot fit curve: Mismatch between K values ({len(k_vals)}) and current residual RMS ({len(i_res_rms)})"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

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

    ax1.scatter(k_vals, v_res_rms, color="crimson", s=45, zorder=5, label="Fundamental Voltage Residual RMS Magnitude ($E_{\\mathrm{rms, V, fund}}$)")
    fit_v_dense = np.polyval(p_v, k_dense)
    ax1.plot(k_dense, fit_v_dense, "--", color="navy", linewidth=2.0, zorder=4, label=f"Fit: ${func_str_v}$ ($R^2={r2_v:.4f}$)")
    ax1.set_xlabel("Number of Simultaneous Co-Events ($K$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Fundamental Voltage Residual RMS Magnitude ($E_{\\mathrm{rms, V, fund}}$) [V]", fontsize=11, fontweight="bold")
    ax1.set_title(f"Fundamental Voltage Residual RMS Magnitude vs. Co-Events ($K = 2 \\dots {max_coevents}$)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2, linestyle=":")
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    ax2.scatter(k_vals, i_res_rms, color="darkgreen", s=45, zorder=5, label="Fundamental Current Residual RMS Magnitude ($E_{\\mathrm{rms, I, fund}}$)")
    fit_i_dense = np.polyval(p_i, k_dense)
    ax2.plot(k_dense, fit_i_dense, "--", color="darkred", linewidth=2.0, zorder=4, label=f"Fit: ${func_str_i}$ ($R^2={r2_i:.4f}$)")
    ax2.set_xlabel("Number of Simultaneous Co-Events ($K$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Fundamental Current Residual RMS Magnitude ($E_{\\mathrm{rms, I, fund}}$) [A]", fontsize=11, fontweight="bold")
    ax2.set_title(f"Fundamental Current Residual RMS Magnitude vs. Co-Events ($K = 2 \\dots {max_coevents}$)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2, linestyle=":")
    ax2.legend(loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, bbox_inches="tight", dpi=300)

    funcs_dict = {"voltage_function": func_str_v, "current_function": func_str_i}
    params_dict = {"voltage_params": (av, bv), "current_params": (ai, bi, ci)}

    return fig, funcs_dict, params_dict


def plot_residual_rms_ratio_vs_coevents(
    max_coevents: int = 25,
    save_path: str = "src/visualization/residual_rms_ratio_vs_coevents.png"
) -> Tuple[plt.Figure, Dict[str, str], Dict[str, Tuple[float, float, float]]]:
    """
    Plots a single plot representing the mean Current Residual RMS / Composed RMS ratio
    averaged across Fundamental and Harmonics channels (excluding slow-varying) vs simultaneous co-events K (K = 2 .. max_coevents).
    Fits quadratic polynomial and saves figure.
    """
    data = compute_residual_rms_vs_coevents(max_coevents=max_coevents)
    k_vals = data["k_values"]
    i_ratios_dict = data["i_ratios_channels"]

    ratio_fund = i_ratios_dict["fundamental"]
    ratio_harm = i_ratios_dict["harmonics"]

    raw_mean_ratios = (ratio_fund + ratio_harm) / 2.0

    for k_val, r_val in zip(k_vals, raw_mean_ratios):
        if r_val >= 1.0:
            print(f"WARNING: Mean current residual RMS ratio {r_val:.4f} at K={int(k_val)} is greater than or equal to 1.0.")

    valid_mask = np.isfinite(raw_mean_ratios)
    k_plot = k_vals[valid_mask]
    ratios_plot = raw_mean_ratios[valid_mask]

    if ratios_plot is None or len(ratios_plot) == 0:
        err_msg = "Cannot fit curve: No valid finite residual RMS ratio values found."
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    p_fit = np.polyfit(k_plot, ratios_plot, 2)
    a_m, b_m, c_m = float(p_fit[0]), float(p_fit[1]), float(p_fit[2])

    fit_vals = np.polyval(p_fit, k_plot)
    ss_res = float(np.sum((ratios_plot - fit_vals) ** 2))
    ss_tot = float(np.sum((ratios_plot - np.mean(ratios_plot)) ** 2))
    r2_m = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    b_sign = "+" if b_m >= 0 else "-"
    c_sign = "+" if c_m >= 0 else "-"
    func_str_mean = f"R_{{I, \\mathrm{{mean}}}}(K) = {abs(a_m):.6f}·K² {b_sign} {abs(b_m):.6f}·K {c_sign} {abs(c_m):.6f}"

    k_dense = np.linspace(float(np.min(k_plot)), float(np.max(k_plot)), 200)
    fit_dense = np.polyval(p_fit, k_dense)

    ax.scatter(k_plot, ratios_plot, color="navy", s=50, zorder=5, label="Mean Current Residual/Composed RMS Ratio ($R_{I, \\mathrm{mean}}$)")
    ax.plot(k_dense, fit_dense, "--", color="crimson", linewidth=2.2, zorder=4, label=f"Fit: ${func_str_mean}$ ($R^2={r2_m:.4f}$)")

    ax.set_xlabel("Number of Simultaneous Co-Events ($K$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Current Residual / Composed RMS Ratio ($R_{I, \\mathrm{mean}}$)", fontsize=11, fontweight="bold")
    ax.set_title("Mean Current Residual RMS / Composed RMS Ratio vs. Simultaneous Co-Events ($R_{I, \\mathrm{mean}}$)", fontsize=12, fontweight="bold")

    text_info = f"Derived Raw Mean Ratio Function:\n${func_str_mean}$\nGoodness of Fit ($R^2$): {r2_m:.4f}"
    ax.text(0.05, 0.80, text_info, transform=ax.transAxes, fontsize=10, fontweight="bold", color="darkblue",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="navy", alpha=0.95), zorder=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    fig.tight_layout()

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, bbox_inches="tight", dpi=300)

    funcs_dict = {
        "mean_ratio_function": func_str_mean,
        "current_ratio_function": func_str_mean,
        "fundamental_ratio_function": func_str_mean,
        "harmonics_ratio_function": func_str_mean,
        "slow_varying_ratio_function": func_str_mean,
    }
    params_dict = {"mean_ratio_params": (a_m, b_m, c_m)}

    return fig, funcs_dict, params_dict


def plot_slow_varying_residual_ratio_vs_coevents(
    max_coevents: int = 25,
    save_path: str = "src/visualization/slow_varying_residual_ratio_vs_coevents.png"
) -> Tuple[plt.Figure, Dict[str, str], Dict[str, Tuple[float, float, float]]]:
    """
    Graphs the residual/joint ratio R_slow(K) = RMS(i_joint_slow - i_composed_slow) / RMS(i_joint_slow)
    of the slow-varying (DC) channel vs simultaneous co-events K (K = 2 .. max_coevents) on a separate plot.
    Fits quadratic polynomial and saves figure.
    """
    data = compute_residual_rms_vs_coevents(max_coevents=max_coevents)
    k_vals = data["k_values"]

    # Retrieve pre-computed slow-varying residual RMS / joint RMS ratios
    ratios_slow = data["i_ratios_channels"]["slow_varying"]

    valid_mask = np.isfinite(ratios_slow)
    k_plot = k_vals[valid_mask]
    ratios_plot = ratios_slow[valid_mask]

    if len(ratios_plot) == 0:
        raise ValueError("Cannot fit curve: No valid finite slow-varying ratio values found.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    p_fit = np.polyfit(k_plot, ratios_plot, 2)
    a_s, b_s, c_s = float(p_fit[0]), float(p_fit[1]), float(p_fit[2])

    fit_vals = np.polyval(p_fit, k_plot)
    ss_res = float(np.sum((ratios_plot - fit_vals) ** 2))
    ss_tot = float(np.sum((ratios_plot - np.mean(ratios_plot)) ** 2))
    r2_s = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    b_sign = "+" if b_s >= 0 else "-"
    c_sign = "+" if c_s >= 0 else "-"
    func_str_slow = f"R_{{I, \\mathrm{{slow}}}}(K) = {abs(a_s):.6f}·K² {b_sign} {abs(b_s):.6f}·K {c_sign} {abs(c_s):.6f}"

    k_dense = np.linspace(float(np.min(k_plot)), float(np.max(k_plot)), 200)
    fit_dense = np.polyval(p_fit, k_dense)

    ax.scatter(k_plot, ratios_plot, color="darkorange", s=50, zorder=5, label="Slow-Varying (DC) Residual/Joint RMS Ratio ($R_{I, \\mathrm{slow}}$)")
    ax.plot(k_dense, fit_dense, "--", color="purple", linewidth=2.2, zorder=4, label=f"Fit: ${func_str_slow}$ ($R^2={r2_s:.4f}$)")

    ax.set_xlabel("Number of Simultaneous Co-Events ($K$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Slow-Varying (DC) Residual / Joint RMS Ratio ($R_{I, \\mathrm{slow}}$)", fontsize=11, fontweight="bold")
    ax.set_title("Slow-Varying (DC) Channel Residual RMS / Joint RMS Ratio vs. Simultaneous Co-Events ($R_{I, \\mathrm{slow}}$)", fontsize=12, fontweight="bold")

    text_info = f"Derived Slow-Varying Ratio Function:\n${func_str_slow}$\nGoodness of Fit ($R^2$): {r2_s:.4f}"
    ax.text(0.05, 0.80, text_info, transform=ax.transAxes, fontsize=10, fontweight="bold", color="purple",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="purple", alpha=0.95), zorder=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    fig.tight_layout()

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, bbox_inches="tight", dpi=300)

    funcs_dict = {"slow_varying_ratio_function": func_str_slow}
    params_dict = {"slow_varying_params": (a_s, b_s, c_s)}

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
    print(f"Mean Current Residual RMS Ratio Function: {funcs2['mean_ratio_function']}")

    fig3, funcs3, _ = plot_slow_varying_residual_ratio_vs_coevents()
    print("\nSlow-varying residual ratio vs co-events plot generated successfully!")
    print(f"Slow-Varying Residual RMS Ratio Function: {funcs3['slow_varying_ratio_function']}")
