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

def run_q3_transformer_spec_analysis(dataset_path: Path = Path("src/simulation/dataset_4.csv")) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 4 not found at {dataset_path}. Run src/simulation/dataset.py first.")

    df_4 = pd.read_csv(dataset_path)
    print("--- Running Question 3 Analysis: Transformer Specification Effect (Dataset 4) ---")

    pair_categories = ["load_load", "fault_fault", "load_fault"]
    results = {"per_category": {}}

    f_v_list, p_v_list = [], []

    avg_r, avg_std, dissimilarity = compute_waveform_pearson_stats(df_4)
    results["avg_pearson_corr"] = avg_r
    results["std_pearson_corr"] = avg_std
    results["dissimilarity"] = dissimilarity

    print(f"Waveform Pearson Correlation (Dataset 4): Mean r_bar = {avg_r:.4f}, Std sigma_r = {avg_std:.4f}, Dissimilarity D = {dissimilarity:.4f}")

    if "gt_pair_category" not in df_4.columns or "gt_transformer_spec_id" not in df_4.columns:
        spec_col = "gt_transformer_spec_id" if "gt_transformer_spec_id" in df_4.columns else ("gt_feeder_id" if "gt_feeder_id" in df_4.columns else None)
        if spec_col:
            tx_groups_v = [group["residual_voltage_magnitude"].values for _, group in df_4.groupby(spec_col)]
            f_v, p_v = stats.f_oneway(*tx_groups_v) if len(tx_groups_v) > 1 else (0.0, 1.0)
        else:
            f_v, p_v = 0.0, 1.0
        results["avg_f_stat_voltage"] = float(f_v) if np.isfinite(f_v) else 0.0
        results["avg_p_val_voltage"] = float(p_v) if np.isfinite(p_v) else 1.0
        print(f"Overall Dataset 4 Transformer Spec Effect (N={len(df_4)}): F={results['avg_f_stat_voltage']:.4f}, p={results['avg_p_val_voltage']:.4e}")
        return results

    for cat in pair_categories:
        df_cat = df_4[df_4["gt_pair_category"] == cat]

        tx_groups_v = [group["residual_voltage_magnitude"].values for _, group in df_cat.groupby("gt_transformer_spec_id")]
        tx_groups_i = [group["residual_current_magnitude"].values for _, group in df_cat.groupby("gt_transformer_spec_id")]

        var_v = sum(np.var(g) for g in tx_groups_v) if tx_groups_v else 0.0
        var_i = sum(np.var(g) for g in tx_groups_i) if tx_groups_i else 0.0

        if len(tx_groups_v) > 1 and all(len(g) > 0 for g in tx_groups_v) and var_v > 0:
            f_v, p_v = stats.f_oneway(*tx_groups_v)
        else:
            f_v, p_v = 0.0, 1.0

        if len(tx_groups_i) > 1 and all(len(g) > 0 for g in tx_groups_i) and var_i > 0:
            f_i, p_i = stats.f_oneway(*tx_groups_i)
        else:
            f_i, p_i = 0.0, 1.0

        f_v_list.append(f_v)
        p_v_list.append(p_v)

        spec_means = df_cat.groupby("gt_transformer_spec_id")[["residual_voltage_magnitude", "residual_current_magnitude"]].mean().to_dict(orient="index")

        results["per_category"][cat] = {
            "n_observations": len(df_cat),
            "f_stat_voltage": float(f_v),
            "p_val_voltage": float(p_v),
            "f_stat_current": float(f_i),
            "p_val_current": float(p_i),
            "spec_means": spec_means
        }

        print(f"Pair Category '{cat}' (N={len(df_cat)}):")
        print(f"  Q3 Transformer Spec Effect (Voltage): F = {f_v:.4f}, p = {p_v:.4e}")
        print(f"  Q3 Transformer Spec Effect (Current): F = {f_i:.4f}, p = {p_i:.4e}")
        for spec_id, means in spec_means.items():
            print(f"    - Tx Spec '{spec_id}': V_res = {means['residual_voltage_magnitude']:.6f}, I_res = {means['residual_current_magnitude']:.6f}")
        print()

    results["avg_f_stat_voltage"] = float(np.mean(f_v_list))
    results["avg_p_val_voltage"] = float(np.mean(p_v_list))

    print("--- Summary Q3 Transformer Spec Effect Across All Pair Categories ---")
    print(f"Average F-stat Voltage: {results['avg_f_stat_voltage']:.4f}, p-value: {results['avg_p_val_voltage']:.4e}\n")

    return results

if __name__ == "__main__":
    run_q3_transformer_spec_analysis()
