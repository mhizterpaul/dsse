"""
LV Distribution Network 1 Specification & Implementation
Aligns with docs/specs/lv1: Network ID LV1, 20 buses, 19 branches, 415V/240V, 1.5 MVA.
Registers consumer units and their loads directly to LV Network 1.
"""

from opendssdirect import dss
import numpy as np
from src.power_plant.consumer_registry import ConsumerRegistry

LV1_SPEC = {
    "network_id": "LV1",
    "topology_type": "Radial",
    "num_buses": 20,
    "num_branches": 19,
    "num_phases": 3,
    "v_lv_nominal_v": 415.0,
    "v_ln_nominal_v": 240.0,
    "base_frequency_hz": 50.0,
    "base_power_mva": 1.5,
    "base_voltage_v": 415.0,
    "line_code": "lv1_conductor_150mm2"
}

def generate_lv1_topology(seed: int = 1001) -> dict:
    rng = np.random.default_rng(seed)
    root_bus = "feeder1_sec"
    buses = [root_bus]
    lines = []

    buses.extend(["f1_node1", "f1_node2"])
    lines.append({
        "name": "down_1_1",
        "bus1": root_bus,
        "bus2": "f1_node1",
        "length": 0.05,
        "units": "km",
        "r1": 0.21, "x1": 0.08, "r0": 0.63, "x0": 0.24, "norm_amps": 350.0
    })
    lines.append({
        "name": "down_1_2",
        "bus1": root_bus,
        "bus2": "f1_node2",
        "length": 0.06,
        "units": "km",
        "r1": 0.21, "x1": 0.08, "r0": 0.63, "x0": 0.24, "norm_amps": 350.0
    })

    for i in range(3, 20):
        bus_name = f"f1_node{i}"
        buses.append(bus_name)
        parent_bus = "f1_node1" if (i % 2 == 1) else f"f1_node{i-2}"
        length = round(float(rng.uniform(0.04, 0.09)), 3)
        lines.append({
            "name": f"down_1_{i}",
            "bus1": parent_bus,
            "bus2": bus_name,
            "length": length,
            "units": "km",
            "r1": 0.21, "x1": 0.08, "r0": 0.63, "x0": 0.24, "norm_amps": 350.0
        })

    return {
        "feeder_idx": 1,
        "spec": LV1_SPEC,
        "buses": buses,
        "lines": lines
    }

def register_lv1_consumers(topology: dict = None, seed: int = 1001, registry: ConsumerRegistry = None) -> ConsumerRegistry:
    """
    Registers consumer units and their load circuits directly to LV Network 1.
    """
    if topology is None:
        topology = generate_lv1_topology(seed=seed)

    if registry is None:
        registry = ConsumerRegistry(seed=seed)

    rng = np.random.default_rng(seed)
    feeder_id = "feeder_1"
    for bus in topology.get("buses", []):
        if not bus.endswith("_sec"):
            cid = f"consumer_{feeder_id}_{bus}"
            base_kw = round(float(rng.uniform(3.0, 10.0)), 2)
            registry.register_consumer(
                consumer_id=cid,
                bus_id=bus,
                feeder_id=feeder_id,
                base_kw=base_kw,
                extra_load_probability=0.45
            )
            if rng.random() < 0.20:
                latent_cid = f"latent_{feeder_id}_{bus}"
                latent_kw = round(float(rng.uniform(2.0, 6.0)), 2)
                registry.register_latent_consumer(
                    consumer_id=latent_cid,
                    bus_id=bus,
                    feeder_id=feeder_id,
                    kw=latent_kw
                )

    return registry

def build_lv1_network(topology: dict = None, loads_dict: dict = None, registry: ConsumerRegistry = None, seed: int = 1001):
    if topology is None:
        topology = generate_lv1_topology(seed=seed)

    if registry is None and loads_dict is None:
        registry = register_lv1_consumers(topology=topology, seed=seed)

    dss.run_command(
        f"new linecode.{LV1_SPEC['line_code']} "
        f"nphases=3 r1=0.21 x1=0.08 r0=0.63 x0=0.24 c1=10.0 c0=5.0 units=km normamps=350.0"
    )

    for ln in topology.get("lines", []):
        dss.run_command(
            f"new line.{ln['name']} "
            f"bus1={ln['bus1']} bus2={ln['bus2']} phases=3 "
            f"r1={ln.get('r1', 0.21)} x1={ln.get('x1', 0.08)} "
            f"r0={ln.get('r0', 0.63)} x0={ln.get('x0', 0.24)} "
            f"length={ln.get('length', 0.05)} units=km normamps=350.0"
        )

    # Apply consumer units and their loads registered to LV Network 1
    if registry is not None:
        for unit in registry.get_all_consumers():
            if unit.feeder_id == "feeder_1":
                for ld in unit.loads:
                    dss.run_command(
                        f"new load.{ld.load_id} bus1={unit.bus_id} phases=3 kv=0.415 kw={ld.kw} pf={ld.pf} model=1 status=fixed"
                    )
    elif loads_dict:
        for ld in loads_dict.get("loads", []):
            dss.run_command(
                f"new load.{ld['name']} bus1={ld['bus']} phases=3 kv=0.415 kw={ld['kw']} pf={ld['pf']} model={ld.get('model', 1)}"
            )

    return registry
