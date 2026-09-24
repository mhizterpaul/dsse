from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from src.loads import get_equipment_model


@dataclass
class LoadDefinition:
    load_id: str
    circuit_id: str
    load_type: str = "ac_motor"
    is_extra_load: bool = False


@dataclass
class ConsumerUnit:
    consumer_id: str
    bus_id: str
    feeder_id: str
    service_line_resistance_ohm: float
    service_line_reactance_ohm: float
    assigned_load_class: Optional[str] = None
    loads: List[LoadDefinition] = field(default_factory=list)
    is_metered: bool = False
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
    class weights, extra load assignments dependent on load power rating, and consumer units.
    """
    LOAD_CLASSES = ["residential", "commercial", "industrial", "agricultural"]
    CLASS_WEIGHTS = {
        "residential": 1.0,
        "commercial": 2.2,
        "industrial": 3.5,
        "agricultural": 1.5
    }

    LOAD_CIRCUIT_TYPES = [
        "ac_motor", "dc_motor_inverter", "microwave", "induction_plate",
        "compressor", "audio_amplifier", "ups", "industrial_fan"
    ]

    CLASS_PRIMARY_LOADS = {
        "residential": ["microwave", "induction_plate", "audio_amplifier"],
        "commercial": ["compressor", "ac_motor", "ups"],
        "industrial": ["ac_motor", "industrial_fan", "dc_motor_inverter"],
        "agricultural": ["ac_motor", "compressor"]
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._registered_consumers: Dict[str, ConsumerUnit] = {}
        self._latent_consumers: Dict[str, ConsumerUnit] = {}

    def get_assigned_weight(self, unit) -> float:
        class_id = getattr(unit, "assigned_load_class", None)
        if class_id is None or class_id not in self.CLASS_WEIGHTS:
            raise ValueError(
                f"Consumer unit '{getattr(unit, 'consumer_id', unit)}' is missing or has invalid assigned_load_class: {class_id}"
            )
        base_w = self.CLASS_WEIGHTS[class_id]
        loads = getattr(unit, "loads", [])

        extra_weight = 0.0
        for ld in loads:
            if getattr(ld, "is_extra_load", False):
                eq = get_equipment_model(ld.load_type)
                extra_weight += float(eq.rated_power_kw * 0.05)

        return float(base_w + extra_weight)

    def register_consumer(
        self,
        consumer_id: str,
        bus_id: str,
        feeder_id: str,
        service_line_resistance_ohm: Optional[float] = None,
        service_line_reactance_ohm: Optional[float] = None,
        assigned_load_class: Optional[str] = None,
        is_metered: bool = False,
        extra_load_probability: float = 0.4
    ) -> ConsumerUnit:
        if assigned_load_class is None:
            assigned_load_class = str(self.rng.choice(self.LOAD_CLASSES, p=[0.60, 0.25, 0.10, 0.05]))

        if service_line_resistance_ohm is None or service_line_reactance_ohm is None:
            raise ValueError(f"Service line resistance and reactance must be provided for consumer {consumer_id}")

        primary_types = self.CLASS_PRIMARY_LOADS.get(assigned_load_class, self.LOAD_CIRCUIT_TYPES)
        base_type = str(self.rng.choice(primary_types))
        base_load = LoadDefinition(
            load_id=f"{consumer_id}_load_base",
            circuit_id=f"{consumer_id}_circuit_1",
            load_type=base_type,
            is_extra_load=False
        )
        loads = [base_load]

        if not is_metered:
            num_extra = int(self.rng.choice([0, 1, 2], p=[0.35, 0.45, 0.20]))
            outside_types = [t for t in self.LOAD_CIRCUIT_TYPES if t not in primary_types]
            if not outside_types:
                outside_types = self.LOAD_CIRCUIT_TYPES
            for k in range(num_extra):
                extra_type = str(self.rng.choice(outside_types))
                extra_load = LoadDefinition(
                    load_id=f"{consumer_id}_extra_{k+1}_{extra_type}",
                    circuit_id=f"{consumer_id}_circuit_extra_{k+1}",
                    load_type=extra_type,
                    is_extra_load=True
                )
                loads.append(extra_load)
        elif self.rng.random() < extra_load_probability:
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
            is_metered=is_metered,
            is_latent_unmetered=False,
            service_line_resistance_ohm=service_line_resistance_ohm,
            service_line_reactance_ohm=service_line_reactance_ohm
        )
        self._registered_consumers[consumer_id] = unit
        return unit

    def register_latent_consumer(
        self,
        consumer_id: str,
        bus_id: str,
        feeder_id: str,
        service_line_resistance_ohm: Optional[float] = None,
        service_line_reactance_ohm: Optional[float] = None,
        load_type: str = "ac_motor"
    ) -> ConsumerUnit:
        if service_line_resistance_ohm is None or service_line_reactance_ohm is None:
            raise ValueError(f"Service line resistance and reactance must be provided for latent consumer {consumer_id}")

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
            assigned_load_class=None,
            loads=[load],
            is_latent_unmetered=True,
            service_line_resistance_ohm=service_line_resistance_ohm,
            service_line_reactance_ohm=service_line_reactance_ohm
        )
        self._latent_consumers[consumer_id] = unit
        return unit

    def get_consumer(self, consumer_id: str) -> Optional[ConsumerUnit]:
        return self._registered_consumers.get(consumer_id, self._latent_consumers.get(consumer_id))

    def get_all_consumers(self) -> List[ConsumerUnit]:
        return list(self._registered_consumers.values()) + list(self._latent_consumers.values())

    def get_registered_consumers(self) -> List[ConsumerUnit]:
        return list(self._registered_consumers.values())

    def get_metered_consumers(self) -> List[ConsumerUnit]:
        return [c for c in self._registered_consumers.values() if c.is_metered]

    def get_unmetered_consumers(self) -> List[ConsumerUnit]:
        return [c for c in self._registered_consumers.values() if not c.is_metered]

    def get_latent_consumers(self) -> List[ConsumerUnit]:
        return list(self._latent_consumers.values())

    def get_consumers_by_class(self, load_class: str) -> List[ConsumerUnit]:
        return [c for c in self._registered_consumers.values() if c.assigned_load_class == load_class]

    def get_consumers_with_extra_loads(self) -> List[ConsumerUnit]:
        return [c for c in self.get_all_consumers() if c.has_extra_load]

    def build_registry_from_topology(self, topology: dict) -> Dict[str, ConsumerUnit]:
        topologies = topology.get("topologies", {})
        if topologies:
            for feeder_idx, sub_topo in topologies.items():
                feeder_id = f"feeder_{feeder_idx}"
                bus_line_map = {ln["bus2"]: ln for ln in sub_topo.get("lines", [])}
                for bus in sub_topo.get("buses", []):
                    if not bus.endswith("_sec"):
                        ln_info = bus_line_map.get(bus, {})
                        length = float(ln_info.get("length", 0.05))
                        r1 = float(ln_info.get("r1", 0.21))
                        x1 = float(ln_info.get("x1", 0.08))
                        r_drop = round(r1 * length, 6)
                        x_drop = round(x1 * length, 6)

                        cid = f"consumer_{feeder_id}_{bus}"
                        self.register_consumer(
                            consumer_id=cid,
                            bus_id=bus,
                            feeder_id=feeder_id,
                            service_line_resistance_ohm=r_drop,
                            service_line_reactance_ohm=x_drop,
                            extra_load_probability=0.45
                        )
                        if self.rng.random() < 0.20:
                            latent_cid = f"latent_{feeder_id}_{bus}"
                            latent_type = str(self.rng.choice(self.LOAD_CIRCUIT_TYPES))
                            self.register_latent_consumer(
                                consumer_id=latent_cid,
                                bus_id=bus,
                                feeder_id=feeder_id,
                                service_line_resistance_ohm=r_drop,
                                service_line_reactance_ohm=x_drop,
                                load_type=latent_type
                            )
        else:
            bus_line_map = {ln["bus2"]: ln for ln in topology.get("lines", [])}
            for bus in topology.get("buses", []):
                if not bus.endswith("_sec"):
                    ln_info = bus_line_map.get(bus, {})
                    length = float(ln_info.get("length", 0.05))
                    r1 = float(ln_info.get("r1", 0.21))
                    x1 = float(ln_info.get("x1", 0.08))
                    r_drop = round(r1 * length, 6)
                    x_drop = round(x1 * length, 6)

                    cid = f"consumer_{bus}"
                    self.register_consumer(
                        consumer_id=cid,
                        bus_id=bus,
                        feeder_id="feeder_1",
                        service_line_resistance_ohm=r_drop,
                        service_line_reactance_ohm=x_drop,
                        extra_load_probability=0.45
                    )

        return self._registered_consumers


class ConsumerLoadClassModel:
    CLASS_WEIGHTS = ConsumerRegistry.CLASS_WEIGHTS

    @classmethod
    def compute_expected_weight(cls, unit, registry=None) -> float:
        if registry is not None and hasattr(registry, "get_assigned_weight"):
            return registry.get_assigned_weight(unit)
        class_id = getattr(unit, "assigned_load_class", None)
        if class_id is None or class_id not in cls.CLASS_WEIGHTS:
            raise ValueError(f"Consumer unit '{getattr(unit, 'consumer_id', unit)}' is missing or has invalid assigned_load_class: {class_id}")
        base_w = cls.CLASS_WEIGHTS[class_id]
        loads = getattr(unit, "loads", [])
        extra_weight = 0.0
        for ld in loads:
            if getattr(ld, "is_extra_load", False):
                eq = get_equipment_model(ld.load_type)
                extra_weight += float(eq.rated_power_kw * 0.05)
        return float(base_w + extra_weight)


def create_default_consumer_registry(topology: dict, seed: int = 42) -> ConsumerRegistry:
    registry = ConsumerRegistry(seed=seed)
    registry.build_registry_from_topology(topology)
    return registry
