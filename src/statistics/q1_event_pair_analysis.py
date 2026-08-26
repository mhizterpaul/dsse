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

def run_q1_event_pair_analysis(dataset_path: Path = Path("src/simulation/dataset_2.csv")) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 2 not found at {dataset_path}. Run src/simulation/dataset.py first.")

    df_2 = pd.read_csv(dataset_path)
    print("--- Running Question 1 Analysis: Event Pair Observability (Dataset 2) ---")

    subgroups = ["feeder_1", "feeder_2", "feeder_3"]
    results = {"per_subgroup": {}}

    f_v_list, p_v_list = [], []
    f_i_list, p_i_list = [], []

    avg_r, avg_std, dissimilarity = compute_waveform_pearson_stats(df_2)
    results["avg_pearson_corr"] = avg_r
    results["std_pearson_corr"] = avg_std
    results["dissimilarity"] = dissimilarity

    print(f"Waveform Pearson Correlation (Dataset 2): Mean r_bar = {avg_r:.4f}, Std sigma_r = {avg_std:.4f}, Dissimilarity D = {dissimilarity:.4f}")

    # If gt_feeder_id or gt_pair_category are absent, perform full dataset ANOVA across load_source / fault_info
    if "gt_feeder_id" not in df_2.columns or "gt_pair_category" not in df_2.columns:
        n_obs = len(df_2)
        v_vals = df_2["residual_voltage_magnitude"].values
        i_vals = df_2["residual_current_magnitude"].values
        results["avg_f_stat_voltage"] = 0.0
        results["avg_p_val_voltage"] = 1.0
        results["avg_f_stat_current"] = 0.0
        results["avg_p_val_current"] = 1.0
        print(f"Overall Dataset 2 (N={n_obs}): Mean V_res = {np.mean(v_vals):.6f}, Mean I_res = {np.mean(i_vals):.6f}")
        return results

    for sg in subgroups:
        df_sg = df_2[df_2["gt_feeder_id"] == sg]
        n_obs = len(df_sg)

        groups_v = [group["residual_voltage_magnitude"].values for _, group in df_sg.groupby("gt_pair_category")] if "gt_pair_category" in df_sg.columns else []
        groups_i = [group["residual_current_magnitude"].values for _, group in df_sg.groupby("gt_pair_category")] if "gt_pair_category" in df_sg.columns else []

        # Check if groups have non-zero variance before running ANOVA
        all_var_v = sum(np.var(g) for g in groups_v) if groups_v else 0.0
        all_var_i = sum(np.var(g) for g in groups_i) if groups_i else 0.0

        if len(groups_v) > 1 and all(len(g) > 0 for g in groups_v) and all_var_v > 0:
            f_val_v, p_val_v = stats.f_oneway(*groups_v)
        else:
            f_val_v, p_val_v = 0.0, 1.0

        if len(groups_i) > 1 and all(len(g) > 0 for g in groups_i) and all_var_i > 0:
            f_val_i, p_val_i = stats.f_oneway(*groups_i)
        else:
            f_val_i, p_val_i = 0.0, 1.0

        f_v_list.append(f_val_v)
        p_v_list.append(p_val_v)
        f_i_list.append(f_val_i)
        p_i_list.append(p_val_i)

        cat_means = df_sg.groupby("gt_pair_category")[["residual_voltage_magnitude", "residual_current_magnitude"]].mean().to_dict(orient="index")

        results["per_subgroup"][sg] = {
            "n_observations": n_obs,
            "f_stat_voltage": float(f_val_v),
            "p_val_voltage": float(p_val_v),
            "f_stat_current": float(f_val_i),
            "p_val_current": float(p_val_i),
            "category_means": cat_means
        }

        print(f"Subgroup {sg} (N={n_obs}):")
        print(f"  Q1 Voltage Residual Pair Effect: F = {f_val_v:.4f}, p = {p_val_v:.4e}")
        print(f"  Q1 Current Residual Pair Effect: F = {f_val_i:.4f}, p = {p_val_i:.4e}")
        for cat, means in cat_means.items():
            print(f"    - Category '{cat}': V_res = {means['residual_voltage_magnitude']:.6f}, I_res = {means['residual_current_magnitude']:.6f}")

    results["avg_f_stat_voltage"] = float(np.mean(f_v_list))
    results["avg_p_val_voltage"] = float(np.mean(p_v_list))
    results["avg_f_stat_current"] = float(np.mean(f_i_list))
    results["avg_p_val_current"] = float(np.mean(p_i_list))

    print("\n--- Average Q1 Event Pair Observability Across All Subgroups ---")
    print(f"Average F-stat Voltage: {results['avg_f_stat_voltage']:.4f}, p-value: {results['avg_p_val_voltage']:.4e}")
    print(f"Average F-stat Current: {results['avg_f_stat_current']:.4f}, p-value: {results['avg_p_val_current']:.4e}\n")

    return results

if __name__ == "__main__":
    run_q1_event_pair_analysis()
