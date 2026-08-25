from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class LoadDefinition:
    load_id: str
    circuit_id: str
    kw: float
    pf: float = 0.95
    load_type: str = "base"  # e.g., 'base', 'extra_hvac', 'extra_ev_charger', 'extra_pv_der'
    is_extra_load: bool = False


@dataclass
class ConsumerUnit:
    consumer_id: str
    bus_id: str
    feeder_id: str
    assigned_load_class: str  # e.g., 'residential', 'commercial', 'industrial', 'agricultural'
    loads: List[LoadDefinition] = field(default_factory=list)

    @property
    def load_circuit_ids(self) -> List[str]:
        return [ld.circuit_id for ld in self.loads]

    @property
    def has_extra_load(self) -> bool:
        return any(ld.is_extra_load for ld in self.loads)


class ConsumerRegistry:
    """
    Consumer Registry in simulation module managing assigned load classes,
    multiple loads, and extra loads for consumer units / load circuits
    during network initialization conditions.
    """
    LOAD_CLASSES = ["residential", "commercial", "industrial", "agricultural"]
    EXTRA_LOAD_TYPES = ["extra_hvac", "extra_ev_charger", "extra_heat_pump"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._consumers: Dict[str, ConsumerUnit] = {}

    def register_consumer(
        self,
        consumer_id: str,
        bus_id: str,
        feeder_id: str,
        assigned_load_class: Optional[str] = None,
        base_kw: float = 5.0,
        pf: float = 0.95,
        extra_load_probability: float = 0.4
    ) -> ConsumerUnit:
        if assigned_load_class is None:
            assigned_load_class = str(self.rng.choice(self.LOAD_CLASSES, p=[0.60, 0.25, 0.10, 0.05]))

        # Base primary load
        base_load = LoadDefinition(
            load_id=f"{consumer_id}_load_base",
            circuit_id=f"{consumer_id}_circuit_1",
            kw=base_kw,
            pf=pf,
            load_type="base",
            is_extra_load=False
        )
        loads = [base_load]

        # Assign extra loads during network initialization if triggered by probability
        if self.rng.random() < extra_load_probability:
            extra_type = str(self.rng.choice(self.EXTRA_LOAD_TYPES))
            extra_kw = round(float(self.rng.uniform(2.0, 7.5)), 2)
            extra_load = LoadDefinition(
                load_id=f"{consumer_id}_{extra_type}",
                circuit_id=f"{consumer_id}_circuit_extra",
                kw=extra_kw,
                pf=0.92,
                load_type=extra_type,
                is_extra_load=True
            )
            loads.append(extra_load)

        unit = ConsumerUnit(
            consumer_id=consumer_id,
            bus_id=bus_id,
            feeder_id=feeder_id,
            assigned_load_class=assigned_load_class,
            loads=loads
        )
        self._consumers[consumer_id] = unit
        return unit

    def get_consumer(self, consumer_id: str) -> Optional[ConsumerUnit]:
        return self._consumers.get(consumer_id)

    def get_all_consumers(self) -> List[ConsumerUnit]:
        return list(self._consumers.values())

    def get_consumers_by_class(self, load_class: str) -> List[ConsumerUnit]:
        return [c for c in self._consumers.values() if c.assigned_load_class == load_class]

    def get_consumers_with_extra_loads(self) -> List[ConsumerUnit]:
        return [c for c in self._consumers.values() if c.has_extra_load]

    def build_registry_from_topology(self, topology: dict) -> Dict[str, ConsumerUnit]:
        """
        Builds and populates consumer registry for all consumer buses/load circuits in given LV topology
        with explicit base and extra loads assigned as part of network initialization.
        """
        topologies = topology.get("topologies", {})
        if topologies:
            for feeder_idx, sub_topo in topologies.items():
                feeder_id = f"feeder_{feeder_idx}"
                for bus in sub_topo.get("buses", []):
                    if not bus.endswith("_sec"):
                        cid = f"consumer_{feeder_id}_{bus}"
                        base_kw = round(float(self.rng.uniform(3.0, 10.0)), 2)
                        self.register_consumer(
                            consumer_id=cid,
                            bus_id=bus,
                            feeder_id=feeder_id,
                            base_kw=base_kw,
                            extra_load_probability=0.45
                        )
        else:
            for bus in topology.get("buses", []):
                if not bus.endswith("_sec"):
                    cid = f"consumer_{bus}"
                    base_kw = round(float(self.rng.uniform(3.0, 10.0)), 2)
                    self.register_consumer(
                        consumer_id=cid,
                        bus_id=bus,
                        feeder_id="feeder_1",
                        base_kw=base_kw,
                        extra_load_probability=0.45
                    )

        return self._consumers


def create_default_consumer_registry(topology: dict, seed: int = 42) -> ConsumerRegistry:
    registry = ConsumerRegistry(seed=seed)
    registry.build_registry_from_topology(topology)
    return registry
