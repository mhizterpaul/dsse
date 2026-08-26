import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def compute_waveform_pearson_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    pairs = [
        ("obs_single_event_1_v_phase_a", "obs_single_event_2_v_phase_a"),
        ("obs_single_event_1_v_phase_b", "obs_single_event_2_v_phase_b"),
        ("obs_single_event_1_v_phase_c", "obs_single_event_2_v_phase_c"),
        ("obs_single_event_1_i_phase_a", "obs_single_event_2_i_phase_a"),
        ("obs_single_event_1_i_phase_b", "obs_single_event_2_i_phase_b"),
        ("obs_single_event_1_i_phase_c", "obs_single_event_2_i_phase_c"),
    ]

    r_means = []
    r_stds = []

    for _, row in df.iterrows():
        r_vals = []
        for col1, col2 in pairs:
            if col1 in row and col2 in row:
                v1 = row[col1]
                v2 = row[col2]
                arr1 = np.array(json.loads(v1)) if isinstance(v1, str) else np.array(v1)
                arr2 = np.array(json.loads(v2)) if isinstance(v2, str) else np.array(v2)

                if len(arr1) > 0 and len(arr2) > 0 and np.std(arr1) > 1e-9 and np.std(arr2) > 1e-9:
                    val, _ = stats.pearsonr(arr1, arr2)
                    val = 0.0 if np.isnan(val) else abs(val)
                else:
                    val = 0.0
            else:
                val = 0.0
            r_vals.append(val)

        r_arr = np.asarray(r_vals)
        r_bar = float(np.mean(r_arr))
        r_std = float(np.sqrt(np.sum((r_arr - r_bar) ** 2) / 5.0))
        r_means.append(r_bar)
        r_stds.append(r_std)

    avg_r = float(np.mean(r_means)) if r_means else 0.0
    avg_std = float(np.mean(r_stds)) if r_stds else 0.0
    dissimilarity = float(1.0 - avg_r)
    return avg_r, avg_std, dissimilarity

def run_q2_time_shift_analysis(dataset_path: Path = Path("src/simulation/dataset_3.csv")) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 3 not found at {dataset_path}. Run src/simulation/dataset.py first.")

    df_3 = pd.read_csv(dataset_path)
    print("--- Running Question 2 Analysis: Time Shift Operation Variation (Dataset 3) ---")

    pair_categories = ["load_load", "fault_fault", "load_fault"]
    results = {"per_category": {}}

    bf_stats_list, bf_p_list = [], []

    avg_r, avg_std, dissimilarity = compute_waveform_pearson_stats(df_3)
    results["avg_pearson_corr"] = avg_r
    results["std_pearson_corr"] = avg_std
    results["dissimilarity"] = dissimilarity

    print(f"Waveform Pearson Correlation (Dataset 3): Mean r_bar = {avg_r:.4f}, Std sigma_r = {avg_std:.4f}, Dissimilarity D = {dissimilarity:.4f}")

    if "gt_pair_category" not in df_3.columns:
        group_sim = df_3[df_3["gt_time_offset_s"] == 0.0]
        group_shift = df_3[df_3["gt_time_offset_s"] > 0.0]
        v_sim = group_sim["residual_voltage_magnitude"].values
        v_shift = group_shift["residual_voltage_magnitude"].values
        stat_v, p_v = (stats.levene(v_sim, v_shift, center="median") if len(v_sim) > 0 and len(v_shift) > 0 else (0.0, 1.0))
        results["avg_brown_forsythe_stat"] = float(stat_v) if np.isfinite(stat_v) else 0.0
        results["avg_p_val"] = float(p_v) if np.isfinite(p_v) else 1.0
        print(f"Overall Dataset 3 Time Shift (N_sim={len(v_sim)}, N_shift={len(v_shift)}): Stat={results['avg_brown_forsythe_stat']:.4f}, p={results['avg_p_val']:.4e}")
        return results

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

        if len(v_sim) > 0 and len(v_shift) > 0 and var_v > 1e-9:
            stat_v, p_v = stats.levene(v_sim, v_shift, center="median")
            if not np.isfinite(stat_v):
                stat_v, p_v = 0.0, 1.0
        else:
            stat_v, p_v = 0.0, 1.0

        if len(i_sim) > 0 and len(i_shift) > 0 and var_i > 1e-9:
            stat_i, p_i = stats.levene(i_sim, i_shift, center="median")
            if not np.isfinite(stat_i):
                stat_i, p_i = 0.0, 1.0
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
        print(f"  Brown-Forsythe Test (Voltage): Stat = {stat_v:.4f}, p = {p_v:.4e}")

    results["avg_brown_forsythe_stat"] = float(np.nanmean(bf_stats_list))
    results["avg_p_val"] = float(np.nanmean(bf_p_list))

    print("--- Summary Q2 Time Shift Variation Across All Pair Categories ---")
    print(f"Average Brown-Forsythe Stat: {results['avg_brown_forsythe_stat']:.4f}, p-value: {results['avg_p_val']:.4e}")

    return results

if __name__ == "__main__":
    run_q2_time_shift_analysis()
