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
from src.transient.events import SingleEquipmentSwitchEvent, EquipmentEquipmentCoEvent, MultiEquipmentCoEvent
from src.transient.no_load_decomposition import decompose_load_waveform


def compute_residual_rms_vs_coevents(
    max_coevents: int = 25,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes voltage and current residual RMS magnitudes and channel ratios for K = 2 .. max_coevents
    directly using CoSimulationRunner and ATP-EMTP / OpenDSS pre-event Thévenin state estimation.

    For each K simultaneous event configuration:
      1. Generates single equipment switching responses and joint simultaneous K-event responses.
      2. Computes linear composed co-event reference by summing single equipment responses.
      3. Computes residual waveforms v_res = v_joint - v_composed and i_res = i_joint - i_composed.
      4. Decomposes waveforms into 3 component channels: Fundamental (50 Hz), Harmonics, and Slow-Varying (SSD).
    Raises ValueError if required parameters, equipment models, or channels are missing.
    """
    runner = CoSimulationRunner()
    plant_data = runner.initialize_plant_session(use_baseline_feeder=True, seed=seed)

    # Solve base network operating point from OpenDSS
    op = plant_data["op"] if "op" in plant_data else None
    if op is None:
        from src.power_plant.plant import solve_operating_point
        op = solve_operating_point(runner.dss)

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

    tx_unit_id = "trans1_lv_boundary"

    # Pre-evaluate single equipment switching responses for all load models
    single_responses = {}
    for eq_type in equipment_types:
        ev = SingleEquipmentSwitchEvent(
            equipment_type=eq_type,
            start_time_s=0.02,
            duration_s=0.10,
            target="trans1",
            parameters={}
        )
        try:
            t_s, v_dict, i_dict, _ = runner.measure_transients(
                op=op, event=ev, scenario_id=f"single_{eq_type}", feeder_idx=1, use_baseline_feeder=True
            )
            single_responses[eq_type] = (t_s, v_dict[tx_unit_id], i_dict[tx_unit_id])
        except Exception as e:
            err_msg = f"Failed to measure transient response for equipment '{eq_type}': {e}"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg) from e

    if len(single_responses) < len(equipment_types):
        err_msg = f"Incomplete single responses: expected {len(equipment_types)}, got {len(single_responses)}"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

    # Evaluate for each K simultaneous co-events using CoSimulationRunner transients
    for K_int in range(2, max_coevents + 1):
        K = float(K_int)

        # Clear steady-state loads in OpenDSS to prevent network overloading as K increases towards 25
        runner.dss.run_command("disable Load.*")
        from src.power_plant.plant import solve_operating_point
        op = solve_operating_point(runner.dss)
        selected_eqs = [equipment_types[i % len(equipment_types)] for i in range(K_int)]

        t_sim, v_eq1, i_eq1 = single_responses[selected_eqs[0]]
        fs = 1.0 / (t_sim[1] - t_sim[0])

        i_composed = sum(single_responses[eq][2] for eq in selected_eqs)
        v_composed = sum(single_responses[eq][1] for eq in selected_eqs) / K

        # Construct multi-equipment co-event with ALL K equipment switched simultaneously
        single_events = [
            SingleEquipmentSwitchEvent(eq_type, 0.02, 0.10, "trans1", {})
            for eq_type in selected_eqs
        ]
        co_ev = MultiEquipmentCoEvent(events=single_events)

        t_s, v_dict, i_dict, _ = runner.measure_transients(
            op=op, event=co_ev, scenario_id=f"coevent_K{K_int}", feeder_idx=1, use_baseline_feeder=True
        )

        v_joint = v_dict[tx_unit_id]
        i_joint = i_dict[tx_unit_id]

        v_res = v_joint - v_composed
        i_res = i_joint - i_composed

        decomp_v_joint = decompose_load_waveform(t_sim, v_joint, fs=fs)
        decomp_v_res = decompose_load_waveform(t_sim, v_res, fs=fs)
        decomp_i_joint = decompose_load_waveform(t_sim, i_joint, fs=fs)
        decomp_i_res = decompose_load_waveform(t_sim, i_res, fs=fs)

        channel_keys = {
            "fundamental": "x_fundamental",
            "harmonics": "x_harmonics",
            "slow_varying": "x_ssd"
        }

        for ch_name, decomp_key in channel_keys.items():
            if (
                decomp_key not in decomp_i_joint or decomp_key not in decomp_i_res or
                decomp_key not in decomp_v_joint or decomp_key not in decomp_v_res
            ):
                err_msg = f"Missing component key '{decomp_key}' in decomposition results"
                print(f"ERROR: {err_msg}")
                raise ValueError(err_msg)

            v_co_ch = float(np.sqrt(np.mean(decomp_v_joint[decomp_key] ** 2)))
            v_res_ch = float(np.sqrt(np.mean(decomp_v_res[decomp_key] ** 2)))
            i_co_ch = float(np.sqrt(np.mean(decomp_i_joint[decomp_key] ** 2)))
            i_res_ch = float(np.sqrt(np.mean(decomp_i_res[decomp_key] ** 2)))

            if i_co_ch <= 0:
                err_msg = f"Non-positive co-event RMS magnitude for channel '{ch_name}' at K={K_int}: {i_co_ch}"
                print(f"ERROR: {err_msg}")
                raise ValueError(err_msg)

            ratio_ch = i_res_ch / (i_co_ch + 1e-9)

            res_v_rms_channels[ch_name].append(v_res_ch)
            co_v_rms_channels[ch_name].append(v_co_ch)
            res_i_rms_channels[ch_name].append(i_res_ch)
            co_i_rms_channels[ch_name].append(i_co_ch)
            i_ratios_channels[ch_name].append(ratio_ch)

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
    Plots a single plot representing the mean Current Residual RMS / Co-Event RMS Magnitude ratio
    averaged across the three component channels (Fundamental, Harmonics, Slow-Varying) vs simultaneous co-events K (K = 2 .. max_coevents).
    Fits quadratic polynomial, displays derived ratio function, and saves the figure.
    """
    data = compute_residual_rms_vs_coevents(max_coevents=max_coevents)
    k_vals = data["k_values"]
    i_ratios_dict = data["i_ratios_channels"]

    # Compute mean ratio across the 3 component channels for each K
    ratio_fund = i_ratios_dict["fundamental"]
    ratio_harm = i_ratios_dict["harmonics"]
    ratio_slow = i_ratios_dict["slow_varying"]

    mean_ratios = (ratio_fund + ratio_harm + ratio_slow) / 3.0

    if mean_ratios is None or len(mean_ratios) == 0 or not np.all(np.isfinite(mean_ratios)):
        err_msg = f"Cannot fit curve: Mean residual RMS ratio values are missing or invalid: {mean_ratios}"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

    if len(k_vals) != len(mean_ratios):
        err_msg = f"Cannot fit curve: Mismatch between K values ({len(k_vals)}) and mean ratios ({len(mean_ratios)})"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    p_fit = np.polyfit(k_vals, mean_ratios, 2)
    a_m, b_m, c_m = float(p_fit[0]), float(p_fit[1]), float(p_fit[2])

    fit_vals = np.polyval(p_fit, k_vals)
    ss_res = float(np.sum((mean_ratios - fit_vals) ** 2))
    ss_tot = float(np.sum((mean_ratios - np.mean(mean_ratios)) ** 2))
    r2_m = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    b_sign = "+" if b_m >= 0 else "-"
    c_sign = "+" if c_m >= 0 else "-"
    func_str_mean = f"R_{{I, \\mathrm{{mean}}}}(K) = {abs(a_m):.6f}·K² {b_sign} {abs(b_m):.6f}·K {c_sign} {abs(c_m):.6f}"

    k_dense = np.linspace(2.0, max_coevents, 200)
    fit_dense = np.polyval(p_fit, k_dense)

    ax.scatter(k_vals, mean_ratios, color="navy", s=50, zorder=5, label="Mean Current Residual/Co-Event RMS Ratio ($R_{I, \\mathrm{mean}}$)")
    ax.plot(k_dense, fit_dense, "--", color="crimson", linewidth=2.2, zorder=4, label=f"Fit: ${func_str_mean}$ ($R^2={r2_m:.4f}$)")

    ax.set_xlabel("Number of Simultaneous Co-Events ($K$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Current Residual / Co-Event RMS Ratio ($R_{I, \\mathrm{mean}}$)", fontsize=11, fontweight="bold")
    ax.set_title(f"Mean Current Residual RMS / Co-Event RMS Ratio vs. Simultaneous Co-Events ($K = 2 \\dots {max_coevents}$)", fontsize=12, fontweight="bold")

    text_info = f"Derived Mean Ratio Function:\n${func_str_mean}$\nGoodness of Fit ($R^2$): {r2_m:.4f}"
    ax.text(0.05, 0.80, text_info, transform=ax.transAxes, fontsize=10, fontweight="bold", color="darkblue",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="navy", alpha=0.95), zorder=10)
    ax.set_ylim(bottom=0.0)
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
