"""
LV Distribution Network 3 Specification & Implementation
Aligns with docs/specs/lv3: Network ID LV3, 30 buses, 29 branches, 415V/240V, 2.0 MVA.
Registers consumer units and their loads directly to LV Network 3.
"""

from opendssdirect import dss
import numpy as np
from src.power_plant.consumer_registry import ConsumerRegistry

LV3_SPEC = {
    "network_id": "LV3",
    "topology_type": "Radial",
    "num_buses": 30,
    "num_branches": 29,
    "num_phases": 3,
    "v_lv_nominal_v": 415.0,
    "v_ln_nominal_v": 240.0,
    "base_frequency_hz": 50.0,
    "base_power_mva": 2.0,
    "base_voltage_v": 415.0,
    "line_code": "lv3_conductor_150mm2"
}

def generate_lv3_topology(seed: int = 1003) -> dict:
    rng = np.random.default_rng(seed)
    root_bus = "feeder3_sec"
    buses = [root_bus]
    lines = []

    buses.extend(["f3_node1", "f3_node2"])
    lines.append({
        "name": "down_3_1",
        "bus1": root_bus,
        "bus2": "f3_node1",
        "length": 0.05,
        "units": "km",
        "r1": 0.21, "x1": 0.08, "r0": 0.63, "x0": 0.24, "norm_amps": 350.0
    })
    lines.append({
        "name": "down_3_2",
        "bus1": root_bus,
        "bus2": "f3_node2",
        "length": 0.06,
        "units": "km",
        "r1": 0.21, "x1": 0.08, "r0": 0.63, "x0": 0.24, "norm_amps": 350.0
    })

    for i in range(3, 30):
        bus_name = f"f3_node{i}"
        buses.append(bus_name)
        parent_bus = "f3_node1" if (i % 2 == 1) else f"f3_node{i-2}"
        length = round(float(rng.uniform(0.04, 0.09)), 3)
        lines.append({
            "name": f"down_3_{i}",
            "bus1": parent_bus,
            "bus2": bus_name,
            "length": length,
            "units": "km",
            "r1": 0.21, "x1": 0.08, "r0": 0.63, "x0": 0.24, "norm_amps": 350.0
        })

    return {
        "feeder_idx": 3,
        "spec": LV3_SPEC,
        "buses": buses,
        "lines": lines
    }

def register_lv3_consumers(topology: dict = None, seed: int = 1003, registry: ConsumerRegistry = None) -> ConsumerRegistry:
    """
    Registers consumer units and their load circuits directly to LV Network 3.
    """
    if topology is None:
        topology = generate_lv3_topology(seed=seed)

    if registry is None:
        registry = ConsumerRegistry(seed=seed)

    rng = np.random.default_rng(seed)
    feeder_id = "feeder_3"
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

    return registry

def build_lv3_network(topology: dict = None, loads_dict: dict = None, registry: ConsumerRegistry = None, seed: int = 1003):
    if topology is None:
        topology = generate_lv3_topology(seed=seed)

    if registry is None and loads_dict is None:
        registry = register_lv3_consumers(topology=topology, seed=seed)

    dss.run_command(
        f"new linecode.{LV3_SPEC['line_code']} "
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

    # Apply consumer units and their loads registered to LV Network 3
    if registry is not None:
        for unit in registry.get_all_consumers():
            if unit.feeder_id == "feeder_3":
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
