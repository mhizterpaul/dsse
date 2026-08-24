import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def run_q2_time_shift_analysis(dataset_path: Path = Path("src/simulation/dataset_3.csv")) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 3 not found at {dataset_path}. Run src/simulation/dataset.py first.")

    df_3 = pd.read_csv(dataset_path)
    print("--- Running Question 2 Analysis: Time Shift Operation Variation (Dataset 3) ---")

    pair_categories = ["load_load", "fault_fault", "load_fault"]
    results = {"per_category": {}}

    bf_stats_list, bf_p_list = [], []

    for cat in pair_categories:
        df_cat = df_3[df_3["gt_pair_category"] == cat]

        group_sim = df_cat[df_cat["gt_time_offset_s"] == 0.0]
        group_shift = df_cat[df_cat["gt_time_offset_s"] > 0.0]

        v_sim = group_sim["residual_voltage_magnitude"].values
        v_shift = group_shift["residual_voltage_magnitude"].values
        i_sim = group_sim["residual_current_magnitude"].values
        i_shift = group_shift["residual_current_magnitude"].values

        var_v = np.var(v_sim) + np.var(v_shift) if len(v_sim) > 0 and len(v_shift) > 0 else 0.0
        var_i = np.var(i_sim) + np.var(i_shift) if len(i_sim) > 0 and len(i_shift) > 0 else 0.0

        if len(v_sim) > 0 and len(v_shift) > 0 and var_v > 0:
            stat_v, p_v = stats.levene(v_sim, v_shift, center="median")
        else:
            stat_v, p_v = 0.0, 1.0

        if len(i_sim) > 0 and len(i_shift) > 0 and var_i > 0:
            stat_i, p_i = stats.levene(i_sim, i_shift, center="median")
        else:
            stat_i, p_i = 0.0, 1.0

        bf_stats_list.append(stat_v)
        bf_p_list.append(p_v)

        results["per_category"][cat] = {
            "n_simultaneous": len(group_sim),
            "n_shifted": len(group_shift),
            "mean_v_residual_simultaneous": float(np.mean(v_sim)) if len(v_sim) > 0 else 0.0,
            "mean_v_residual_shifted": float(np.mean(v_shift)) if len(v_shift) > 0 else 0.0,
            "brown_forsythe_stat_voltage": float(stat_v),
            "p_val_voltage": float(p_v),
            "brown_forsythe_stat_current": float(stat_i),
            "p_val_current": float(p_i)
        }

        print(f"Pair Category '{cat}':")
        print(f"  Simultaneous (N={len(group_sim)}): V_res = {np.mean(v_sim):.6f}, I_res = {np.mean(i_sim):.6f}")
        print(f"  Time-Shifted (N={len(group_shift)}): V_res = {np.mean(v_shift):.6f}, I_res = {np.mean(i_shift):.6f}")
        print(f"  Brown-Forsythe Test (Voltage): Stat = {stat_v:.4f}, p = {p_v:.4e}\n")

    results["avg_brown_forsythe_stat"] = float(np.mean(bf_stats_list))
    results["avg_p_val"] = float(np.mean(bf_p_list))

    print("--- Summary Q2 Time Shift Variation Across All Pair Categories ---")
    print(f"Average Brown-Forsythe Stat: {results['avg_brown_forsythe_stat']:.4f}, p-value: {results['avg_p_val']:.4e}\n")

    return results

if __name__ == "__main__":
    run_q2_time_shift_analysis()
