import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def compute_waveform_pearson_stats(df: pd.DataFrame, snr_db: float = 35.0) -> dict:
    pairs = [
        ("obs_single_event_1_v_phase_a", "obs_single_event_2_v_phase_a"),
        ("obs_single_event_1_v_phase_b", "obs_single_event_2_v_phase_b"),
        ("obs_single_event_1_v_phase_c", "obs_single_event_2_v_phase_c"),
        ("obs_single_event_1_i_phase_a", "obs_single_event_2_i_phase_a"),
        ("obs_single_event_1_i_phase_b", "obs_single_event_2_i_phase_b"),
        ("obs_single_event_1_i_phase_c", "obs_single_event_2_i_phase_c"),
    ]

    eta_noise = 10.0 ** (-snr_db / 20.0)
    col_correlations = [[] for _ in pairs]

    for _, row in df.iterrows():
        for p_idx, (col1, col2) in enumerate(pairs):
            if col1 in row and col2 in row:
                v1 = row[col1]
                v2 = row[col2]
                arr1 = np.array(json.loads(v1)) if isinstance(v1, str) else np.array(v1)
                arr2 = np.array(json.loads(v2)) if isinstance(v2, str) else np.array(v2)

                if len(arr1) > 0 and len(arr2) > 0 and np.std(arr1) > 1e-9 and np.std(arr2) > 1e-9:
                    val, _ = stats.pearsonr(arr1, arr2)
                    raw_val = 0.0 if np.isnan(val) else abs(val)
                    val = max(0.0, raw_val - eta_noise)
                else:
                    val = 0.0
            else:
                val = 0.0
            col_correlations[p_idx].append(val)

    col_corr_vector = [float(np.mean(c_list)) if c_list else 0.0 for c_list in col_correlations]
    mean_waveform_correlation = float(np.mean(col_corr_vector)) if col_corr_vector else 0.0
    corr_std = float(np.std(col_corr_vector)) if col_corr_vector else 0.0

    return {
        "col_corr_vector": col_corr_vector,
        "mean_waveform_correlation": mean_waveform_correlation,
        "corr_std": corr_std
    }


def run_q3_transformer_spec_analysis(dataset_path: Path = Path("src/simulation/dataset_4.csv")) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 4 not found at {dataset_path}. Run src/simulation/dataset.py first.")

    df_4 = pd.read_csv(dataset_path)
    print("--- Running Question 3 Analysis: Transformer Specification Effect (Dataset 4) ---")

    stats_res = compute_waveform_pearson_stats(df_4)
    col_corr_vector = stats_res["col_corr_vector"]
    mean_waveform_correlation = stats_res["mean_waveform_correlation"]
    corr_std = stats_res["corr_std"]

    print(f"Waveform Pearson Correlation (Dataset 4): Mean Correlation = {mean_waveform_correlation:.4f}, Std = {corr_std:.4f}")
    print(f"  Per-Column Correlation Vector: {[round(c, 4) for c in col_corr_vector]}")

    v_vals = df_4["residual_voltage_magnitude"].values
    i_vals = df_4["residual_current_magnitude"].values
    print(f"Overall Dataset 4 Transformer Spec Effect (N={len(df_4)}): Mean V_res = {np.mean(v_vals):.6f}, Mean I_res = {np.mean(i_vals):.6f}")

    return {
        "col_corr_vector": col_corr_vector,
        "mean_waveform_correlation": mean_waveform_correlation,
        "corr_std": corr_std,
        "mean_v_residual": float(np.mean(v_vals)),
        "mean_i_residual": float(np.mean(i_vals))
    }


if __name__ == "__main__":
    run_q3_transformer_spec_analysis()
