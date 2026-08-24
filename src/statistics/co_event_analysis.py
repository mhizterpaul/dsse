import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

def run_dataset_3_coevent_analysis():
    """
    Dataset 3 Co-Event Residual Variation Analysis (Q2 & Q3).
    Uses Brown-Forsythe test on residual voltage and current magnitudes across:
      1. Simultaneous vs Time-Shifted co-events (Q2).
      2. Line fault types (LG, LL, LLG, LLL) in equipment+fault co-events (Q3).
    """
    data_path = Path(__file__).parent.parent / "simulation" / "dataset_3.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset 3 CSV not found at {data_path}. Run dataset generation first.")

    df = pd.read_csv(data_path)

    print("--- Dataset 3 Co-Event Residual Variation Analysis (Q2 & Q3) ---")

    # 1. Q2: Simultaneous vs Time-Shifted Analysis
    simultaneous_mask = df["gt_time_offset_s"] == 0.0
    group_sim = df[simultaneous_mask]["residual_current_magnitude"].values
    group_shift = df[~simultaneous_mask]["residual_current_magnitude"].values

    if len(group_sim) > 1 and len(group_shift) > 1 and (np.std(group_sim) > 1e-12 or np.std(group_shift) > 1e-12):
        stat_time, p_time = stats.levene(group_sim, group_shift, center='median')
    else:
        stat_time, p_time = 0.0, 1.0

    print(f"Q2: Timing Effect (Simultaneous vs Time-Shifted):")
    print(f"  Brown-Forsythe Statistic = {stat_time:.4f}, p-value = {p_time:.4e}")

    # 2. Q3: Equipment + Line Fault Co-Events across Fault Types (LG, LL, LLG, LLL)
    eq_fault_df = df[df["gt_coevent_class"] == "equipment_line_fault_coevent"]
    fault_groups = [g["residual_current_magnitude"].values for _, g in eq_fault_df.groupby("gt_event_2_fault_type") if len(g) > 1]

    valid_fault_groups = [fg for fg in fault_groups if np.std(fg) > 1e-12]

    if len(valid_fault_groups) > 1:
        stat_fault, p_fault = stats.levene(*valid_fault_groups, center='median')
    else:
        stat_fault, p_fault = 0.0, 1.0

    print(f"\nQ3: Line Fault Effect on Equipment Observability (LG, LL, LLG, LLL):")
    print(f"  Brown-Forsythe Statistic = {stat_fault:.4f}, p-value = {p_fault:.4e}")

    return {
        "stat_timing": float(stat_time),
        "p_timing": float(p_time),
        "stat_fault_type": float(stat_fault),
        "p_fault_type": float(p_fault)
    }

if __name__ == "__main__":
    run_dataset_3_coevent_analysis()
