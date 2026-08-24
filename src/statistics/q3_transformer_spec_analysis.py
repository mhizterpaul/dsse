import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def run_q3_transformer_spec_analysis(dataset_path: Path = Path("src/simulation/dataset_4.csv")) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 4 not found at {dataset_path}. Run src/simulation/dataset.py first.")

    df_4 = pd.read_csv(dataset_path)
    print("--- Running Question 3 Analysis: Transformer Specification Effect (Dataset 4) ---")

    pair_categories = ["load_load", "fault_fault", "load_fault"]
    results = {"per_category": {}}

    f_v_list, p_v_list = [], []

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
