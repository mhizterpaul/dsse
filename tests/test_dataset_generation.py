import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.dataset import generate_experiments_dataset


def test_dataset_generation_only():
    print("========================================================================")
    print("EVOKING dataset.py (ONLY) via tests/test_dataset_generation.py...")
    print("========================================================================")

    dataset_1, dataset_2, dataset_3, dataset_4 = generate_experiments_dataset(n_scenarios=2, write_to_disk=True)

    print("\n[SUCCESS] Datasets generated and persisted successfully via dataset.py!")
    print(f"Dataset 1 shape: {dataset_1.shape}")
    print(f"Dataset 2 shape: {dataset_2.shape}")
    print(f"Dataset 3 shape: {dataset_3.shape}")
    print(f"Dataset 4 shape: {dataset_4.shape}\n")

    print("--- DATASET 1 HEAD ---")
    print(dataset_1[['gt_scenario_id', 'gt_feeder_id', 'gt_total_consumer_energy_kwh', 'est_time_adjusted_cla_unsampled_energy_kwh']].head(5))

    print("\n--- DATASET 2 HEAD ---")
    print(dataset_2[['gt_scenario_id', 'gt_pair_category', 'residual_voltage_magnitude', 'residual_current_magnitude']].head(5))

    print("\n--- DATASET 3 HEAD ---")
    print(dataset_3[['gt_scenario_id', 'gt_pair_category', 'gt_time_offset_s', 'residual_voltage_magnitude']].head(5))

    print("\n--- DATASET 4 HEAD ---")
    print(dataset_4[['gt_scenario_id', 'gt_transformer_spec_id', 'residual_voltage_magnitude', 'residual_current_magnitude']].head(5))

    # Inspect generated ATP output files (.lis, .pl4) and CSV datasets in codebase
    atp_cases_dir = PROJECT_ROOT / "src" / "simulation" / "atp_cases"
    atp_outputs = list(atp_cases_dir.glob("case_*.*"))
    print("\nPersisted ATP outputs in src/simulation/atp_cases/:")
    for f in sorted(atp_outputs):
        print(f"  - {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    test_dataset_generation_only()
