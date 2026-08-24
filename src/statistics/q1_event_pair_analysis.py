import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def run_q1_event_pair_analysis(dataset_path: Path = Path("src/simulation/dataset_2.csv")) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 2 not found at {dataset_path}. Run src/simulation/dataset.py first.")

    df_2 = pd.read_csv(dataset_path)
    print("--- Running Question 1 Analysis: Event Pair Observability (Dataset 2) ---")

    subgroups = ["feeder_1", "feeder_2", "feeder_3"]
    results = {"per_subgroup": {}}

    f_v_list, p_v_list = [], []
    f_i_list, p_i_list = [], []

    for sg in subgroups:
        df_sg = df_2[df_2["gt_feeder_id"] == sg]
        n_obs = len(df_sg)

        groups_v = [group["residual_voltage_magnitude"].values for _, group in df_sg.groupby("gt_pair_category")]
        groups_i = [group["residual_current_magnitude"].values for _, group in df_sg.groupby("gt_pair_category")]

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
