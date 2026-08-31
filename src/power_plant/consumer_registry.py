from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

# Physical service line resistance constant for LV service drops (Ohms)
DEFAULT_SERVICE_LINE_RESISTANCE_OHM = 0.05


@dataclass
class LoadDefinition:
    load_id: str
    circuit_id: str
    load_type: str = "ac_motor"  # e.g., 'ac_motor', 'dc_motor_inverter', 'microwave', 'induction_plate', 'compressor', 'audio_amplifier', 'ups', 'industrial_fan'
    is_extra_load: bool = False


@dataclass
class ConsumerUnit:
    consumer_id: str
    bus_id: str
    feeder_id: str
    assigned_load_class: Optional[str] = None  # None for latent/unmetered hidden consumers
    loads: List[LoadDefinition] = field(default_factory=list)
    is_latent_unmetered: bool = False
    service_line_resistance_ohm: float = DEFAULT_SERVICE_LINE_RESISTANCE_OHM

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
    LOAD_CIRCUIT_TYPES = [
        "ac_motor", "dc_motor_inverter", "microwave", "induction_plate",
        "compressor", "audio_amplifier", "ups", "industrial_fan"
    ]
    DEFAULT_SERVICE_LINE_RESISTANCE = DEFAULT_SERVICE_LINE_RESISTANCE_OHM

    def __init__(self, seed: int = 42, service_line_resistance_ohm: float = DEFAULT_SERVICE_LINE_RESISTANCE_OHM):
        self.rng = np.random.default_rng(seed)
        self.service_line_resistance_ohm = service_line_resistance_ohm
        self._registered_consumers: Dict[str, ConsumerUnit] = {}
        self._latent_consumers: Dict[str, ConsumerUnit] = {}

    def register_consumer(
        self,
        consumer_id: str,
        bus_id: str,
        feeder_id: str,
        assigned_load_class: Optional[str] = None,
        extra_load_probability: float = 0.4,
        service_line_resistance_ohm: Optional[float] = None
    ) -> ConsumerUnit:
        if assigned_load_class is None:
            assigned_load_class = str(self.rng.choice(self.LOAD_CLASSES, p=[0.60, 0.25, 0.10, 0.05]))

        if service_line_resistance_ohm is None:
            service_line_resistance_ohm = self.service_line_resistance_ohm

        base_type = str(self.rng.choice(self.LOAD_CIRCUIT_TYPES))
        base_load = LoadDefinition(
            load_id=f"{consumer_id}_load_base",
            circuit_id=f"{consumer_id}_circuit_1",
            load_type=base_type,
            is_extra_load=False
        )
        loads = [base_load]

        if self.rng.random() < extra_load_probability:
            extra_type = str(self.rng.choice(self.LOAD_CIRCUIT_TYPES))
            extra_load = LoadDefinition(
                load_id=f"{consumer_id}_{extra_type}",
                circuit_id=f"{consumer_id}_circuit_extra",
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
            is_latent_unmetered=False,
            service_line_resistance_ohm=service_line_resistance_ohm
        )
        self._registered_consumers[consumer_id] = unit
        return unit

    def register_latent_consumer(
        self,
        consumer_id: str,
        bus_id: str,
        feeder_id: str,
        load_type: str = "ac_motor",
        service_line_resistance_ohm: Optional[float] = None
    ) -> ConsumerUnit:
        """
        Registers a hidden/latent consumer unit without an assigned class in the LV network.
        Used for evaluating non-technical losses (NTL) and theft estimation error.
        """
        if service_line_resistance_ohm is None:
            service_line_resistance_ohm = self.service_line_resistance_ohm

        load = LoadDefinition(
            load_id=f"{consumer_id}_latent_load",
            circuit_id=f"{consumer_id}_latent_circuit",
            load_type=load_type,
            is_extra_load=True
        )
        unit = ConsumerUnit(
            consumer_id=consumer_id,
            bus_id=bus_id,
            feeder_id=feeder_id,
            assigned_load_class=None,  # No assigned class for latent/unmetered units
            loads=[load],
            is_latent_unmetered=True,
            service_line_resistance_ohm=service_line_resistance_ohm
        )
        self._latent_consumers[consumer_id] = unit
        return unit

    def get_consumer(self, consumer_id: str) -> Optional[ConsumerUnit]:
        return self._registered_consumers.get(consumer_id, self._latent_consumers.get(consumer_id))

    def get_all_consumers(self) -> List[ConsumerUnit]:
        return list(self._registered_consumers.values()) + list(self._latent_consumers.values())

    def get_registered_consumers(self) -> List[ConsumerUnit]:
        return list(self._registered_consumers.values())

    def get_metered_consumers(self, sampling_fraction: float = 0.36) -> List[ConsumerUnit]:
        """
        Returns the subset of registered consumer units that are metered/sampled based on sampling fraction.
        """
        registered = self.get_registered_consumers()
        num_metered = int(len(registered) * sampling_fraction)
        return registered[:num_metered]

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
                        self.register_consumer(
                            consumer_id=cid,
                            bus_id=bus,
                            feeder_id=feeder_id,
                            extra_load_probability=0.45
                        )
                        if self.rng.random() < 0.20:
                            latent_cid = f"latent_{feeder_id}_{bus}"
                            latent_type = str(self.rng.choice(self.LOAD_CIRCUIT_TYPES))
                            self.register_latent_consumer(
                                consumer_id=latent_cid,
                                bus_id=bus,
                                feeder_id=feeder_id,
                                load_type=latent_type
                            )
        else:
            for bus in topology.get("buses", []):
                if not bus.endswith("_sec"):
                    cid = f"consumer_{bus}"
                    self.register_consumer(
                        consumer_id=cid,
                        bus_id=bus,
                        feeder_id="feeder_1",
                        extra_load_probability=0.45
                    )

        return self._registered_consumers


def create_default_consumer_registry(topology: dict, seed: int = 42) -> ConsumerRegistry:
    registry = ConsumerRegistry(seed=seed)
    registry.build_registry_from_topology(topology)
    return registry
