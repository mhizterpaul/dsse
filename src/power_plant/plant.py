import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from src.power_plant.sources import configure_generator, apply_generator_profile
from power_plant.lv_transformers import get_distribution_transformer_spec
from src.power_plant.lv_network_1 import generate_lv1_topology, build_lv1_network, register_lv1_consumers
from src.power_plant.lv_network_2 import generate_lv2_topology, build_lv2_network, register_lv2_consumers
from src.power_plant.lv_network_3 import generate_lv3_topology, build_lv3_network, register_lv3_consumers
from src.power_plant.consumer_registry import ConsumerRegistry





def generate_known_radial_topology(feeder_idx: int, num_buses: int = 20, rng=None, seed: int = 42) -> dict:
    """
    Generates deterministic known radial tree topologies (LV1, LV2, LV3) aligned with docs/specs.
    """
    if feeder_idx == 1:
        return generate_lv1_topology(seed=1000 + seed + feeder_idx)
    elif feeder_idx == 2:
        return generate_lv2_topology(seed=1000 + seed + feeder_idx)
    elif feeder_idx == 3:
        return generate_lv3_topology(seed=1000 + seed + feeder_idx)
    else:
        return generate_lv1_topology(seed=1000 + seed + feeder_idx)


generate_radial_topology = generate_known_radial_topology


def identify_candidate_consumer_units(topology: dict) -> list[dict]:
    """
    Identifies candidate consumer consumer_units and edge transformer consumer_units across the known LV network.
    """
    candidate_consumer_units = []

    topologies = topology.get("topologies", {})
    if topologies:
        for f_id, sub_topo in topologies.items():
            for ln in sub_topo.get("lines", []):
                parent = ln["bus1"]
                child = ln["bus2"]
                line_name = ln["name"]

                candidate_consumer_units.append({
                    "consumer_unit_id": f"consumer_unit_{line_name}",
                    "bus": child,
                    "parent_bus": parent,
                    "branch_id": line_name,
                    "branch_type": "consumer_line",
                    "consumer_eligible": True
                })
    else:
        for ln in topology.get("lines", []):
            parent = ln["bus1"]
            child = ln["bus2"]
            line_name = ln["name"]

            candidate_consumer_units.append({
                "consumer_unit_id": f"consumer_unit_{line_name}",
                "bus": child,
                "parent_bus": parent,
                "branch_id": line_name,
                "branch_type": "consumer_line",
                "consumer_eligible": True
            })

    # LV secondary terminals of the distribution transformers (Feeder boundary consumer_units)
    for idx in [1, 2, 3]:
        candidate_consumer_units.append({
            "consumer_unit_id": f"trans{idx}_lv_boundary_consumer_unit",
            "bus": f"feeder{idx}_sec",
            "parent_bus": f"feeder{idx}_head",
            "branch_id": f"transformer.trans{idx}",
            "branch_type": "transformer_boundary",
            "consumer_eligible": True
        })

    return candidate_consumer_units


identify_candidate_consumer_units = identify_candidate_consumer_units


def select_consumer_units(candidate_consumer_units: list[dict], fraction: float, seed: int) -> list[dict]:
    """
    Selects transformer boundary consumer_units and a configured fraction of consumer consumer_units.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"consumer_fraction must be in (0.0, 1.0], got {fraction}")

    transformer_units = [m for m in candidate_consumer_units if m.get("branch_type") == "transformer_boundary"]
    consumer_units = [m for m in candidate_consumer_units if m.get("branch_type") != "transformer_boundary"]

    n_consumer_units = max(1, int(np.ceil(fraction * len(consumer_units)))) if consumer_units else 0

    rng = np.random.default_rng(seed)

    if consumer_units:
        selected_indices = rng.choice(len(consumer_units), size=n_consumer_units, replace=False)
        selected_consumer_units = [consumer_units[i] for i in selected_indices]
    else:
        selected_consumer_units = []

    return transformer_units + selected_consumer_units


select_selected_consumer_units = select_consumer_units


@dataclass
class OperatingPoint:
    time_s: float
    generator_p_kw: float
    generator_q_kvar: float
    feeder_p_kw: dict
    feeder_q_kvar: dict
    transformer_loading: dict
    voltage_pu: dict
    frequency_hz: float
    transient_waveforms: Optional[object] = None
    phase_voltages_v: Optional[dict] = None
    phase_angles_deg: Optional[dict] = None

    def import_atp_cases(self, atp_waveforms):
        self.transient_waveforms = atp_waveforms


def initialize_known_plant(use_baseline_transformers: bool = False, topology: dict = None, seed: int = 42) -> ConsumerRegistry:
    """
    Initializes the fixed upstream distribution station and registers downstream consumers on each LV network.
    """
    print("INFO: Initializing OpenDSS Physics-Based Known Plant Model (33/11/0.415 kV)...")

    

    configure_generator(p_kw=1500.0, q_kvar=0.0)

    

    for f_id in [1, 2, 3]:
        spec = get_distribution_transformer_spec(f_id, use_baseline=use_baseline_transformers)
        dss.run_command(
            f"new transformer.{spec['name']} "
            f"phases={spec['phases']} windings={spec['windings']} "
            f"buses=[{','.join(spec['buses'])}] "
            f"conns=[{','.join(spec['conns'])}] "
            f"kvs=[{','.join(map(str, spec['kvs']))}] "
            f"kvas=[{','.join(map(str, spec['kvas']))}] "
            f"%r={spec['r_pct']} "
            f"xhl={spec['xhl_pct']} "
            f"%noloadloss={spec.get('noloadloss_pct', 0.1)} "
            f"%imag={spec.get('imag_pct', 0.8)}"
        )

    if topology is None:
        top1 = generate_lv1_topology(seed=seed + 1)
        top2 = generate_lv2_topology(seed=seed + 2)
        top3 = generate_lv3_topology(seed=seed + 3)
        topology = {"topologies": {1: top1, 2: top2, 3: top3}}

    registry = ConsumerRegistry(seed=seed)
    topologies = topology.get("topologies", {})

    register_lv1_consumers(topology=topologies.get(1), seed=seed + 1, registry=registry)
    if 2 in topologies:
        register_lv2_consumers(topology=topologies.get(2), seed=seed + 2, registry=registry)
    if 3 in topologies:
        register_lv3_consumers(topology=topologies.get(3), seed=seed + 3, registry=registry)

    print("INFO: OpenDSS Known Plant Model and Consumer Registry successfully initialized.")
    return registry


def build_single_lv_network_composition(
    feeder_idx: int = 1,
    generator_p_kw: float = 1500.0,
    generator_q_kvar: float = 0.0,
    use_baseline_transformers: bool = True,
    loads_dict: dict = None,
    seed: int = 42
) -> dict:
    """
    Composes Case 1: Single LV network configuration (1 LV feeder network).
    """
    top = generate_known_radial_topology(feeder_idx, seed=seed)
    single_topology = {"topologies": {feeder_idx: top}}

    registry = initialize_known_plant(use_baseline_transformers=use_baseline_transformers, topology=single_topology, seed=seed)

    if generator_p_kw > 0:
        configure_generator(p_kw=generator_p_kw, q_kvar=generator_q_kvar)

    if feeder_idx == 1:
        build_lv1_network(topology=top, loads_dict=loads_dict, registry=registry, seed=seed)
    elif feeder_idx == 2:
        build_lv2_network(topology=top, loads_dict=loads_dict, registry=registry, seed=seed)
    else:
        build_lv3_network(topology=top, loads_dict=loads_dict, registry=registry, seed=seed)

    dss.run_command("solve")
    candidate_consumer_units = identify_candidate_consumer_units(single_topology)

    return {
        "topology": single_topology,
        "registry": registry,
        "candidate_consumer_units": candidate_consumer_units
    }


def build_three_lv_networks_composition(
    generator_p_kw: float = 1500.0,
    generator_q_kvar: float = 0.0,
    use_baseline_transformers: bool = True,
    loads_dict: dict = None,
    seed: int = 42
) -> dict:
    """
    Composes Case 2: Three LV networks configuration (LV1, LV2, LV3 networks).
    """
    top1 = generate_lv1_topology(seed=seed + 1)
    top2 = generate_lv2_topology(seed=seed + 2)
    top3 = generate_lv3_topology(seed=seed + 3)

    combined_topology = {
        "topologies": {
            1: top1,
            2: top2,
            3: top3
        }
    }

    registry = initialize_known_plant(use_baseline_transformers=use_baseline_transformers, topology=combined_topology, seed=seed)

    if generator_p_kw > 0:
        configure_generator(p_kw=generator_p_kw, q_kvar=generator_q_kvar)

    build_lv1_network(topology=top1, loads_dict=loads_dict, registry=registry, seed=seed + 1)
    build_lv2_network(topology=top2, loads_dict=loads_dict, registry=registry, seed=seed + 2)
    build_lv3_network(topology=top3, loads_dict=loads_dict, registry=registry, seed=seed + 3)

    dss.run_command("solve")
    candidate_consumer_units = identify_candidate_consumer_units(combined_topology)

    return {
        "topology": combined_topology,
        "registry": registry,
        "candidate_consumer_units": candidate_consumer_units
    }


build_power_plant_and_downstream_networks = build_three_lv_networks_composition


def solve_operating_point(p_kw: float, q_kvar: float, time_s: float = 0.0) -> OperatingPoint:
    """
    Applies generator profiles, runs OpenDSS power flow, and extracts electrical operating point.
    """
    apply_generator_profile(p_kw, q_kvar)

    

    feeder_p = {}
    feeder_q = {}
    loading = {}
    voltage_pu = {}
    phase_voltages_v = {}
    phase_angles_deg = {}

    for idx in [1, 2, 3]:
        consumer_unit = {
            "consumer_unit_id": f"trans{idx}_lv_boundary_consumer_unit",
            "boundary_unit_id": f"trans{idx}_lv_pcc",
            "bus": f"feeder{idx}_sec",
            "parent_bus": f"feeder{idx}_head",
            "branch_id": f"transformer.trans{idx}",
            "branch_type": "transformer"
        }
        

        



    return OperatingPoint(
        time_s=time_s,
        generator_p_kw=p_kw,
        generator_q_kvar=q_kvar,
        feeder_p_kw=feeder_p,
        feeder_q_kvar=feeder_q,
        transformer_loading=loading,
        voltage_pu=voltage_pu,
        phase_voltages_v=phase_voltages_v,
        phase_angles_deg=phase_angles_deg
    )
