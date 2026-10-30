import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def compute_waveform_pearson_stats(df: pd.DataFrame) -> dict:
    pairs = [
        ("obs_single_event_1_v_phase_a", "obs_single_event_2_v_phase_a"),
        ("obs_single_event_1_v_phase_b", "obs_single_event_2_v_phase_b"),
        ("obs_single_event_1_v_phase_c", "obs_single_event_2_v_phase_c"),
        ("obs_single_event_1_i_phase_a", "obs_single_event_2_i_phase_a"),
        ("obs_single_event_1_i_phase_b", "obs_single_event_2_i_phase_b"),
        ("obs_single_event_1_i_phase_c", "obs_single_event_2_i_phase_c"),
        ("obs_single_event_1_vi_phase_diff_phase_a", "obs_single_event_2_vi_phase_diff_phase_a"),
        ("obs_single_event_1_vi_phase_diff_phase_b", "obs_single_event_2_vi_phase_diff_phase_b"),
        ("obs_single_event_1_vi_phase_diff_phase_c", "obs_single_event_2_vi_phase_diff_phase_c"),
    ]

    target_channels = ["fundamental", "harmonics", "slow_varying", "phase_angle"]
    channels = [c for c in target_channels if c in df["channel"].values] if "channel" in df.columns else [None]

    all_corrs = []
    channel_correlations = {}

    for ch in channels:
        ch_df = df[df["channel"] == ch] if ch is not None else df
        if len(ch_df) == 0:
            continue

        col_correlations = [[] for _ in pairs]

        for idx, row in ch_df.iterrows():
            for p_idx, (col1, col2) in enumerate(pairs):
                if col1 not in row or col2 not in row:
                    err_msg = f"Missing required waveform columns '{col1}' or '{col2}' in dataset row {idx}"
                    print(f"ERROR: {err_msg}")
                    raise ValueError(err_msg)

                v1 = row[col1]
                v2 = row[col2]
                arr1 = np.array(json.loads(v1)) if isinstance(v1, str) else np.array(v1)
                arr2 = np.array(json.loads(v2)) if isinstance(v2, str) else np.array(v2)

                if len(arr1) == 0 or len(arr2) == 0:
                    err_msg = f"Empty waveform array in column '{col1}' or '{col2}' at row {idx}"
                    print(f"ERROR: {err_msg}")
                    raise ValueError(err_msg)

                std1 = np.std(arr1)
                std2 = np.std(arr2)
                if std1 <= 0 or std2 <= 0:
                    val_calc = 0.0
                else:
                    val_calc, _ = stats.pearsonr(arr1, arr2)
                    if np.isnan(val_calc):
                        val_calc = 0.0

                val = abs(val_calc)
                col_correlations[p_idx].append(val)

        ch_corr_vector = [float(np.mean(c_list)) for c_list in col_correlations]
        all_corrs.extend(ch_corr_vector)
        if ch is not None:
            channel_correlations[ch] = float(np.mean(ch_corr_vector))

    d_fund = 1.0 - channel_correlations.get("fundamental", 0.0)
    d_harm = 1.0 - channel_correlations.get("harmonics", 0.0)
    d_dc = 1.0 - channel_correlations.get("slow_varying", 0.0)
    d_phase = 1.0 - channel_correlations.get("phase_angle", 0.0)

    d_list = [d_fund, d_harm, d_dc, d_phase]
    d_mean = float(np.mean(d_list))
    d_std = float(np.std(d_list))

    mean_waveform_correlation = float(np.mean(all_corrs)) if all_corrs else 0.0
    corr_std = float(np.std(all_corrs)) if all_corrs else 0.0

    return {
        "mean_waveform_correlation": mean_waveform_correlation,
        "corr_std": corr_std,
        "channel_correlations": channel_correlations,
        "d_fund": d_fund,
        "d_harm": d_harm,
        "d_dc": d_dc,
        "d_phase": d_phase,
        "d_mean": d_mean,
        "d_std": d_std
    }


def run_q3_transformer_spec_analysis(dataset_path: Path = Path("src/simulation/dataset_4.csv")) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 4 not found at {dataset_path}. Run src/simulation/dataset.py first.")

    df_4 = pd.read_csv(dataset_path)
    print("--- Running Question 3 Analysis: Transformer Specification Effect (Dataset 4) ---")

    stats_res = compute_waveform_pearson_stats(df_4)
    mean_waveform_correlation = stats_res["mean_waveform_correlation"]
    corr_std = stats_res["corr_std"]

    print(f"Waveform Pearson Correlation across channels (Dataset 4): Mean Correlation = {mean_waveform_correlation:.4f}, Std = {corr_std:.4f}")
    print(f"  Channel Dissimilarities: Fundamental={stats_res['d_fund']:.4f}, Harmonics={stats_res['d_harm']:.4f}, DC={stats_res['d_dc']:.4f}, Phase={stats_res['d_phase']:.4f}")
    print(f"  Overall Mean Dissimilarity = {stats_res['d_mean']:.4f}, Dissimilarity Std = {stats_res['d_std']:.4f}")

    v_vals = df_4["residual_voltage_magnitude"].values
    i_vals = df_4["residual_current_magnitude"].values
    print(f"Overall Dataset 4 Transformer Spec Effect (N={len(df_4)}): Mean V_res = {np.nanmean(v_vals):.6f}, Mean I_res = {np.nanmean(i_vals):.6f}")

    return {
        "mean_waveform_correlation": mean_waveform_correlation,
        "corr_std": corr_std,
        "mean_v_residual": float(np.nanmean(v_vals)),
        "mean_i_residual": float(np.nanmean(i_vals)),
        "d_fund": stats_res["d_fund"],
        "d_harm": stats_res["d_harm"],
        "d_dc": stats_res["d_dc"],
        "d_phase": stats_res["d_phase"],
        "d_mean": stats_res["d_mean"],
        "d_std": stats_res["d_std"]
    }


if __name__ == "__main__":
    run_q3_transformer_spec_analysis()
