import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.dataset import generate_experiments_dataset
from src.power_plant.consumer_registry import ConsumerRegistry, create_default_consumer_registry
from src.power_plant.plant import generate_known_radial_topology

def test_consumer_registry():
    print("Testing ConsumerRegistry load class assignments...")
    topologies = {f: generate_known_radial_topology(f, 15) for f in [1, 2, 3]}
    registry = create_default_consumer_registry({"topologies": topologies})
    consumers = registry.get_all_consumers()

    assert len(consumers) > 0, "ConsumerRegistry should register consumers for topology buses"
    print(f"Registered {len(consumers)} consumer units across LV network feeders.")

    classes = set(c.assigned_load_class for c in consumers)
    print(f"Assigned load classes across consumer units: {classes}")

    res_consumers = registry.get_consumers_by_class("residential")
    print(f"Residential consumer count: {len(res_consumers)}")
    print("[SUCCESS] ConsumerRegistry verified successfully!\n")

def test_dataset_generation():
    print("========================================================================")
    print("Evoking dataset.py via tests/test_dataset_generation.py...")
    print("========================================================================")

    dataset_1, dataset_2, dataset_3, dataset_4 = generate_experiments_dataset(n_scenarios=2, write_to_disk=True)

    print("\n[SUCCESS] Datasets generated successfully!")
    print(f"Dataset 1 shape: {dataset_1.shape}")
    print(f"Dataset 2 shape: {dataset_2.shape}")
    print(f"Dataset 3 shape: {dataset_3.shape}")
    print(f"Dataset 4 shape: {dataset_4.shape}\n")

    print("--- DATASET 1 HEAD ---")
    print(dataset_1[['gt_scenario_id', 'gt_feeder_id', 'gt_total_consumer_energy_kwh', 'est_time_adjusted_cla_unmetered_energy_kwh']].head(5))

    print("\n--- DATASET 2 HEAD ---")
    print(dataset_2[['gt_scenario_id', 'gt_pair_category', 'residual_voltage_magnitude', 'residual_current_magnitude']].head(5))

    print("\n--- DATASET 3 HEAD ---")
    print(dataset_3[['gt_scenario_id', 'gt_pair_category', 'gt_time_offset_s', 'residual_voltage_magnitude']].head(5))

    print("\n--- DATASET 4 HEAD ---")
    print(dataset_4[['gt_scenario_id', 'gt_transformer_spec_id', 'residual_voltage_magnitude', 'residual_current_magnitude']].head(5))

if __name__ == "__main__":
    test_consumer_registry()
    test_dataset_generation()
