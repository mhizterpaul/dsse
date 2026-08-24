from dataclasses import dataclass
from typing import Optional
from src.transient.events import TransientEvent

@dataclass
class EMTEvent:
    event_id: str
    event_type: str
    start_time_s: float
    duration_s: Optional[float]
    target_element: Optional[str]
    target_bus: Optional[str]
    phase_mask: Optional[tuple[bool, bool, bool]]
    parameters: dict

@dataclass
class NetworkRealization:
    realization_id: str
    buses: list[str]
    lines: list[dict]
    transformers: list[dict]
    loads: list[dict]
    capacitors: list[dict]
    motors: list[dict]
    ders: list[dict]
    source: dict
    metered_consumers: list[dict]
    latent_state: dict
    channel_map: dict

@dataclass
class KnownLVNetworkScenario:
    scenario_id: str
    num_buses: int
    num_lines: int
    topology: dict
    line_parameters: dict
    loads: dict
    load_composition: dict
    motor_penetration: float
    capacitor_configuration: dict
    transformer_loading: dict
    switching_events: list

# Alias for backward compatibility
HiddenNetworkScenario = KnownLVNetworkScenario

@dataclass
class SimulationScenario:
    known_network: KnownLVNetworkScenario
    generator_p_kw: float
    generator_q_kvar: float
    events: list[EMTEvent]
    meter_fraction: float = 0.36
    seed: int = 42

    def __init__(
        self,
        known_network: KnownLVNetworkScenario = None,
        generator_p_kw: float = 1500.0,
        generator_q_kvar: float = 0.0,
        events: list = None,
        meter_fraction: float = 0.36,
        seed: int = 42,
        hidden_network: KnownLVNetworkScenario = None
    ):
        self.known_network = known_network if known_network is not None else hidden_network
        self.generator_p_kw = generator_p_kw
        self.generator_q_kvar = generator_q_kvar
        self.events = events if events is not None else []
        self.meter_fraction = meter_fraction
        self.seed = seed
        if not (0.0 < self.meter_fraction <= 1.0):
            raise ValueError(f"meter_fraction must be in (0.0, 1.0], got {self.meter_fraction}")

    @property
    def hidden_network(self) -> KnownLVNetworkScenario:
        return self.known_network
