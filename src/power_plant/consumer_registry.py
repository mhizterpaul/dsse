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
    assigned_load_class: Optional[str] = None  # None for latent/unmetered hidden consumers
    loads: List[LoadDefinition] = field(default_factory=list)
    is_latent_unmetered: bool = False

    @property
    def load_circuit_ids(self) -> List[str]:
        return [ld.circuit_id for ld in self.loads]

    @property
    def has_extra_load(self) -> bool:
        return any(ld.is_extra_load for ld in self.loads)


class ConsumerRegistry:
    """
    Consumer Registry in power_plant module managing assigned load classes,
    registered consumer units, and hidden/latent consumer units without assigned classes
    for calculating non-technical losses as specified in paper.md.
    """
    LOAD_CLASSES = ["residential", "commercial", "industrial", "agricultural"]
    EXTRA_LOAD_TYPES = ["extra_hvac", "extra_ev_charger", "extra_heat_pump"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._registered_consumers: Dict[str, ConsumerUnit] = {}
        self._latent_consumers: Dict[str, ConsumerUnit] = {}

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

        base_load = LoadDefinition(
            load_id=f"{consumer_id}_load_base",
            circuit_id=f"{consumer_id}_circuit_1",
            kw=base_kw,
            pf=pf,
            load_type="base",
            is_extra_load=False
        )
        loads = [base_load]

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
            loads=loads,
            is_latent_unmetered=False
        )
        self._registered_consumers[consumer_id] = unit
        return unit

    def register_latent_consumer(
        self,
        consumer_id: str,
        bus_id: str,
        feeder_id: str,
        kw: float = 4.0,
        pf: float = 0.95
    ) -> ConsumerUnit:
        """
        Registers a hidden/latent consumer unit without an assigned class in the LV network.
        Used for evaluating non-technical losses (NTL) and theft estimation error.
        """
        load = LoadDefinition(
            load_id=f"{consumer_id}_latent_load",
            circuit_id=f"{consumer_id}_latent_circuit",
            kw=kw,
            pf=pf,
            load_type="latent_unmetered",
            is_extra_load=True
        )
        unit = ConsumerUnit(
            consumer_id=consumer_id,
            bus_id=bus_id,
            feeder_id=feeder_id,
            assigned_load_class=None,  # No assigned class for latent/unmetered units
            loads=[load],
            is_latent_unmetered=True
        )
        self._latent_consumers[consumer_id] = unit
        return unit

    def get_consumer(self, consumer_id: str) -> Optional[ConsumerUnit]:
        return self._registered_consumers.get(consumer_id, self._latent_consumers.get(consumer_id))

    def get_all_consumers(self) -> List[ConsumerUnit]:
        return list(self._registered_consumers.values()) + list(self._latent_consumers.values())

    def get_registered_consumers(self) -> List[ConsumerUnit]:
        return list(self._registered_consumers.values())

    def get_latent_consumers(self) -> List[ConsumerUnit]:
        return list(self._latent_consumers.values())

    def get_consumers_by_class(self, load_class: str) -> List[ConsumerUnit]:
        return [c for c in self._registered_consumers.values() if c.assigned_load_class == load_class]

    def get_consumers_with_extra_loads(self) -> List[ConsumerUnit]:
        return [c for c in self.get_all_consumers() if c.has_extra_load]

    def build_registry_from_topology(self, topology: dict) -> Dict[str, ConsumerUnit]:
        """
        Builds and populates consumer registry for all consumer buses/load circuits in given LV topology,
        including registered consumer units and hidden/latent consumer units for NTL evaluation.
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
                        # Add latent/hidden consumer unit for 15% of nodes
                        if self.rng.random() < 0.15:
                            latent_cid = f"latent_{feeder_id}_{bus}"
                            latent_kw = round(float(self.rng.uniform(2.0, 6.0)), 2)
                            self.register_latent_consumer(
                                consumer_id=latent_cid,
                                bus_id=bus,
                                feeder_id=feeder_id,
                                kw=latent_kw
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

        return self._registered_consumers


def create_default_consumer_registry(topology: dict, seed: int = 42) -> ConsumerRegistry:
    registry = ConsumerRegistry(seed=seed)
    registry.build_registry_from_topology(topology)
    return registry
