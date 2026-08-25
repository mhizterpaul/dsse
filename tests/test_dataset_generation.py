import sys
import os
import glob
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner
from src.simulation.dataset import generate_experiments_dataset
from src.power_plant.consumer_registry import ConsumerRegistry, create_default_consumer_registry
from src.power_plant.plant import generate_known_radial_topology
from src.transient.events import SingleEquipmentSwitchEvent, SingleLineFaultEvent


def test_runner_and_atp():
    print("========================================================================")
    print("TESTING CoSimulationRunner & ATP-EMTP (.lis / .pl4) GENERATION & PARSING")
    print("========================================================================")

    topologies = {f: generate_known_radial_topology(f, 15) for f in [1, 2, 3]}
    registry = create_default_consumer_registry({"topologies": topologies})
    candidate_meters = [
        {"meter_id": f"trans{f}_lv_boundary_meter", "branch_type": "transformer_boundary"} for f in [1, 2, 3]
    ]

    runner = CoSimulationRunner()
    ev = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, "f1_node1", {})

    loads_dist = {
        "loads": [{"name": ld.load_id, "bus": unit.bus_id, "kw": ld.kw, "pf": ld.pf} for unit in registry.get_all_consumers() for ld in unit.loads],
        "capacitors": [],
        "motors": [],
        "ders": []
    }

    print("Running CoSimulationRunner.run_simulation...")
    sim_res = runner.run_simulation(
        topology={"topologies": topologies},
        loads=loads_dist,
        events=[ev],
        generator_p_kw=1500.0,
        generator_q_kvar=0.0,
        meter_fraction=0.36,
        use_baseline_transformers=True,
        include_load_event=True,
        include_fault_event=False,
        scenario_id="runner_test_case",
        seed=42
    )

    print("\nSimulationResult Summary:")
    print(f"  - Time points count: {len(sim_res.time_s)}")
    print(f"  - Metered consumers count: {len(sim_res.metered_consumers)}")
    print(f"  - Processed meters count: {len(sim_res.processed_meters)}")
    print(f"  - Consumer load transients count: {len(sim_res.consumer_load_transients)}")
    print(f"  - Transformer transients count: {len(sim_res.transformer_transients)}")

    # Check generated ATP files (.ATP, .lis, .dbg, .pl4)
    atp_cases_dir = PROJECT_ROOT / "src" / "simulation" / "atp_cases"
    atp_files = list(atp_cases_dir.glob("case_ac_motor_switch.*"))
    print("\nGenerated ATP case files in src/simulation/atp_cases/:")
    for f in sorted(atp_files):
        print(f"  - {f.name} ({f.stat().st_size} bytes)")

    print("[SUCCESS] CoSimulationRunner and ATP execution verified successfully!\n")


def test_dataset_generation():
    print("========================================================================")
    print("EVOKING dataset.py via tests/test_dataset_generation.py...")
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
    test_runner_and_atp()
    test_dataset_generation()
