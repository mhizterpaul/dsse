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
        ("obs_single_event_1_vi_phase_diff_phase_a", "obs_single_event_2_vi_phase_diff_phase_a"),
        ("obs_single_event_1_vi_phase_diff_phase_b", "obs_single_event_2_vi_phase_diff_phase_b"),
        ("obs_single_event_1_vi_phase_diff_phase_c", "obs_single_event_2_vi_phase_diff_phase_c"),
    ]

    eta_noise = 10.0 ** (-snr_db / 20.0)
    target_channels = ["fundamental", "harmonics", "slow_varying", "phase_angle"]
    channels = [c for c in target_channels if c in df["channel"].values] if "channel" in df.columns else [None]

    all_corrs = []
    discounts = []
    channel_correlations = {}

    for ch in channels:
        ch_df = df[df["channel"] == ch] if ch is not None else df
        if len(ch_df) == 0:
            continue

        res_mags = []
        comp_mags = []
        for _, row in ch_df.iterrows():
            v_a = np.array(json.loads(row["obs_composed_event_v_phase_a"])) if isinstance(row["obs_composed_event_v_phase_a"], str) else np.array(row["obs_composed_event_v_phase_a"])
            v_b = np.array(json.loads(row["obs_composed_event_v_phase_b"])) if isinstance(row["obs_composed_event_v_phase_b"], str) else np.array(row["obs_composed_event_v_phase_b"])
            v_c = np.array(json.loads(row["obs_composed_event_v_phase_c"])) if isinstance(row["obs_composed_event_v_phase_c"], str) else np.array(row["obs_composed_event_v_phase_c"])

            i_a = np.array(json.loads(row["obs_composed_event_i_phase_a"])) if isinstance(row["obs_composed_event_i_phase_a"], str) else np.array(row["obs_composed_event_i_phase_a"])
            i_b = np.array(json.loads(row["obs_composed_event_i_phase_b"])) if isinstance(row["obs_composed_event_i_phase_b"], str) else np.array(row["obs_composed_event_i_phase_b"])
            i_c = np.array(json.loads(row["obs_composed_event_i_phase_c"])) if isinstance(row["obs_composed_event_i_phase_c"], str) else np.array(row["obs_composed_event_i_phase_c"])

            comp_v = np.sqrt(np.mean(v_a**2 + v_b**2 + v_c**2))
            comp_i = np.sqrt(np.mean(i_a**2 + i_b**2 + i_c**2))
            comp_mags.append(comp_v + comp_i)

            rv_raw = row.get("residual_voltage_magnitude", 0.0)
            ri_raw = row.get("residual_current_magnitude", 0.0)
            res_v = float(rv_raw) if pd.notna(rv_raw) else 0.0
            res_i = float(ri_raw) if pd.notna(ri_raw) else 0.0
            res_mags.append(res_v + res_i)

        if len(res_mags) == 0 or len(comp_mags) == 0:
            err_msg = f"Empty residual or composed magnitude lists for channel '{ch}'"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

        avg_res_mag = float(np.mean(res_mags))
        avg_comp_mag = float(np.mean(comp_mags))

        if avg_comp_mag <= 0:
            err_msg = f"Non-positive average composed magnitude for channel '{ch}': {avg_comp_mag}"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

        mag_ratio = avg_res_mag / avg_comp_mag

        discount_factor = eta_noise + mag_ratio
        discounts.append(discount_factor)

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

                raw_val = abs(val_calc)
                val = max(0.0, raw_val - discount_factor)
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
    avg_discount = float(np.mean(discounts)) if discounts else 0.0

    return {
        "mean_waveform_correlation": mean_waveform_correlation,
        "corr_std": corr_std,
        "discount_factor": avg_discount,
        "channel_correlations": channel_correlations,
        "d_fund": d_fund,
        "d_harm": d_harm,
        "d_dc": d_dc,
        "d_phase": d_phase,
        "d_mean": d_mean,
        "d_std": d_std
    }


def run_q2_time_shift_analysis(dataset_path: Path = Path("src/simulation/dataset_3.csv")) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 3 not found at {dataset_path}. Run src/simulation/dataset.py first.")

    df_3 = pd.read_csv(dataset_path)
    print("--- Running Question 2 Analysis: Time Shift Operation Variation (Dataset 3) ---")

    stats_res = compute_waveform_pearson_stats(df_3)
    mean_waveform_correlation = stats_res["mean_waveform_correlation"]
    corr_std = stats_res["corr_std"]
    discount_factor = stats_res["discount_factor"]

    print(f"Waveform Pearson Correlation across channels (Dataset 3): Mean Correlation = {mean_waveform_correlation:.4f}, Std = {corr_std:.4f}")
    print(f"  Channel Dissimilarities: Fundamental={stats_res['d_fund']:.4f}, Harmonics={stats_res['d_harm']:.4f}, DC={stats_res['d_dc']:.4f}, Phase={stats_res['d_phase']:.4f}")
    print(f"  Overall Mean Dissimilarity = {stats_res['d_mean']:.4f}, Dissimilarity Std = {stats_res['d_std']:.4f}")

    group_sim = df_3[df_3["gt_time_offset_s"] == 0.0]
    group_shift = df_3[df_3["gt_time_offset_s"] > 0.0]

    v_sim = group_sim["residual_voltage_magnitude"].values
    v_shift = group_shift["residual_voltage_magnitude"].values
    i_sim = group_sim["residual_current_magnitude"].values
    i_shift = group_shift["residual_current_magnitude"].values

    mean_v_sim = float(np.nanmean(v_sim)) if len(v_sim) > 0 else 0.0
    mean_i_sim = float(np.nanmean(i_sim)) if len(i_sim) > 0 else 0.0
    mean_v_shift = float(np.nanmean(v_shift)) if len(v_shift) > 0 else 0.0
    mean_i_shift = float(np.nanmean(i_shift)) if len(i_shift) > 0 else 0.0

    print(f"Overall Dataset 3 Time Shift (N_sim={len(group_sim)}, N_shift={len(group_shift)}):")
    print(f"  Simultaneous: Mean V_res = {mean_v_sim:.6f}, Mean I_res = {mean_i_sim:.6f}")
    print(f"  Time-Shifted: Mean V_res = {mean_v_shift:.6f}, Mean I_res = {mean_i_shift:.6f}")

    return {
        "mean_waveform_correlation": mean_waveform_correlation,
        "corr_std": corr_std,
        "discount_factor": discount_factor,
        "mean_v_residual_simultaneous": mean_v_sim,
        "mean_v_residual_shifted": mean_v_shift,
        "d_fund": stats_res["d_fund"],
        "d_harm": stats_res["d_harm"],
        "d_dc": stats_res["d_dc"],
        "d_phase": stats_res["d_phase"],
        "d_mean": stats_res["d_mean"],
        "d_std": stats_res["d_std"]
    }


if __name__ == "__main__":
    run_q2_time_shift_analysis()
