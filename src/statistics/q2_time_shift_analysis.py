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

    # Compute average scalar residual and composed magnitudes to form the magnitude penalty ratio
    res_mags = []
    comp_mags = []
    for _, row in df.iterrows():
        v_a = np.array(json.loads(row["obs_composed_event_v_phase_a"])) if isinstance(row["obs_composed_event_v_phase_a"], str) else np.array(row["obs_composed_event_v_phase_a"])
        v_b = np.array(json.loads(row["obs_composed_event_v_phase_b"])) if isinstance(row["obs_composed_event_v_phase_b"], str) else np.array(row["obs_composed_event_v_phase_b"])
        v_c = np.array(json.loads(row["obs_composed_event_v_phase_c"])) if isinstance(row["obs_composed_event_v_phase_c"], str) else np.array(row["obs_composed_event_v_phase_c"])

        i_a = np.array(json.loads(row["obs_composed_event_i_phase_a"])) if isinstance(row["obs_composed_event_i_phase_a"], str) else np.array(row["obs_composed_event_i_phase_a"])
        i_b = np.array(json.loads(row["obs_composed_event_i_phase_b"])) if isinstance(row["obs_composed_event_i_phase_b"], str) else np.array(row["obs_composed_event_i_phase_b"])
        i_c = np.array(json.loads(row["obs_composed_event_i_phase_c"])) if isinstance(row["obs_composed_event_i_phase_c"], str) else np.array(row["obs_composed_event_i_phase_c"])

        comp_v = np.sqrt(np.mean(v_a**2 + v_b**2 + v_c**2))
        comp_i = np.sqrt(np.mean(i_a**2 + i_b**2 + i_c**2))
        comp_mags.append(comp_v + comp_i)

        res_v = float(row.get("residual_voltage_magnitude", 0.0))
        res_i = float(row.get("residual_current_magnitude", 0.0))
        res_mags.append(res_v + res_i)

    avg_res_mag = float(np.mean(res_mags)) if res_mags else 0.0
    avg_comp_mag = float(np.mean(comp_mags)) if comp_mags else 1.0
    mag_ratio = avg_res_mag / (avg_comp_mag + 1e-9)

    discount_factor = eta_noise + mag_ratio
    col_correlations = [[] for _ in pairs]

    for _, row in df.iterrows():
        for p_idx, (col1, col2) in enumerate(pairs):
            val = 0.0
            if col1 in row and col2 in row:
                v1 = row[col1]
                v2 = row[col2]
                arr1 = np.array(json.loads(v1)) if isinstance(v1, str) else np.array(v1)
                arr2 = np.array(json.loads(v2)) if isinstance(v2, str) else np.array(v2)

                if len(arr1) > 0 and len(arr2) > 0 and np.std(arr1) > 1e-9 and np.std(arr2) > 1e-9:
                    val_calc, _ = stats.pearsonr(arr1, arr2)
                    raw_val = 0.0 if np.isnan(val_calc) else abs(val_calc)
                    val = max(0.0, raw_val - discount_factor)

            col_correlations[p_idx].append(val)

    col_corr_vector = [float(np.mean(c_list)) if c_list else 0.0 for c_list in col_correlations]
    mean_waveform_correlation = float(np.mean(col_corr_vector)) if col_corr_vector else 0.0
    corr_std = float(np.std(col_corr_vector)) if col_corr_vector else 0.0

    return {
        "col_corr_vector": col_corr_vector,
        "mean_waveform_correlation": mean_waveform_correlation,
        "corr_std": corr_std,
        "avg_res_mag": avg_res_mag,
        "avg_comp_mag": avg_comp_mag,
        "mag_ratio": mag_ratio,
        "discount_factor": discount_factor
    }


def run_q2_time_shift_analysis(dataset_path: Path = Path("src/simulation/dataset_3.csv")) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 3 not found at {dataset_path}. Run src/simulation/dataset.py first.")

    df_3 = pd.read_csv(dataset_path)
    print("--- Running Question 2 Analysis: Time Shift Operation Variation (Dataset 3) ---")

    stats_res = compute_waveform_pearson_stats(df_3)
    col_corr_vector = stats_res["col_corr_vector"]
    mean_waveform_correlation = stats_res["mean_waveform_correlation"]
    corr_std = stats_res["corr_std"]
    mag_ratio = stats_res["mag_ratio"]
    discount_factor = stats_res["discount_factor"]

    print(f"Waveform Pearson Correlation (Dataset 3): Mean Correlation = {mean_waveform_correlation:.4f}, Std = {corr_std:.4f}")
    print(f"  Residual/Composed Magnitude Ratio: {mag_ratio:.4f}, Total Discount Factor = {discount_factor:.4f}")
    print(f"  Per-Column Correlation Vector: {[round(c, 4) for c in col_corr_vector]}")

    group_sim = df_3[df_3["gt_time_offset_s"] == 0.0]
    group_shift = df_3[df_3["gt_time_offset_s"] > 0.0]

    v_sim = group_sim["residual_voltage_magnitude"].values
    v_shift = group_shift["residual_voltage_magnitude"].values
    i_sim = group_sim["residual_current_magnitude"].values
    i_shift = group_shift["residual_current_magnitude"].values

    print(f"Overall Dataset 3 Time Shift (N_sim={len(group_sim)}, N_shift={len(group_shift)}):")
    print(f"  Simultaneous: Mean V_res = {np.mean(v_sim):.6f}, Mean I_res = {np.mean(i_sim):.6f}")
    print(f"  Time-Shifted: Mean V_res = {np.mean(v_shift):.6f}, Mean I_res = {np.mean(i_shift):.6f}")

    return {
        "col_corr_vector": col_corr_vector,
        "mean_waveform_correlation": mean_waveform_correlation,
        "corr_std": corr_std,
        "mag_ratio": mag_ratio,
        "discount_factor": discount_factor,
        "mean_v_residual_simultaneous": float(np.mean(v_sim)) if len(v_sim) > 0 else 0.0,
        "mean_v_residual_shifted": float(np.mean(v_shift)) if len(v_shift) > 0 else 0.0
    }


if __name__ == "__main__":
    run_q2_time_shift_analysis()
