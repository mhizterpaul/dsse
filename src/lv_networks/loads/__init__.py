from src.lv_networks.loads.base import EquipmentCircuit
from src.lv_networks.loads.ac_motor import get_ac_motor
from src.lv_networks.loads.dc_motor_inverter import get_dc_motor_inverter
from src.lv_networks.loads.microwave import get_microwave
from src.lv_networks.loads.induction_plate import get_induction_plate
from src.lv_networks.loads.compressor import get_compressor
from src.lv_networks.loads.audio_amplifier import get_audio_amplifier
from src.lv_networks.loads.ups import get_ups
from src.lv_networks.loads.industrial_fan import get_industrial_fan
import numpy as np

EQUIPMENT_REGISTRY = {
    "ac_motor": get_ac_motor,
    "dc_motor_inverter": get_dc_motor_inverter,
    "microwave": get_microwave,
    "induction_plate": get_induction_plate,
    "compressor": get_compressor,
    "audio_amplifier": get_audio_amplifier,
    "ups": get_ups,
    "industrial_fan": get_industrial_fan
}

def get_equipment_model(equipment_type: str, rated_power_kw: float = None) -> EquipmentCircuit:
    if equipment_type not in EQUIPMENT_REGISTRY:
        raise ValueError(f"Unknown equipment type '{equipment_type}'. Supported types: {list(EQUIPMENT_REGISTRY.keys())}")

    factory = EQUIPMENT_REGISTRY[equipment_type]
    if rated_power_kw is not None:
        return factory(rated_power_kw=rated_power_kw)
    return factory()

def distribute_loads(buses: list, rng=None) -> dict:
    """
    Distributes loads of different classes across the hidden buses.
    Uses local seeded RNG for perfect reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    loads = []
    capacitors = []
    motors = []
    ders = []

    for bus in buses[1:]:
        if rng.random() < 0.6:
            load_kw = rng.uniform(5.0, 25.0)
            l_model = int(rng.choice([1, 2, 3]))
            pf = float(rng.choice([0.85, 0.90, 0.95]))
            loads.append({
                "name": f"l_{bus}",
                "bus": bus,
                "kw": round(load_kw, 2),
                "pf": pf,
                "model": l_model
            })

        if rng.random() < 0.12:
            cap_kvar = float(rng.choice([15.0, 30.0, 45.0]))
            capacitors.append({
                "name": f"c_{bus}",
                "bus": bus,
                "kvar": cap_kvar
            })

        if rng.random() < 0.08:
            motors.append({
                "name": f"m_{bus}",
                "bus": bus,
                "kw": round(float(rng.uniform(10.0, 30.0)), 1),
                "pf": 0.8
            })

        if rng.random() < 0.05:
            ders.append({
                "name": f"der_{bus}",
                "bus": bus,
                "kw": round(float(rng.uniform(5.0, 20.0)), 1)
            })

    return {
        "loads": loads,
        "capacitors": capacitors,
        "motors": motors,
        "ders": ders
    }
