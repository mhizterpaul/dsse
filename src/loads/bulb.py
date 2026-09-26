try:
    from src.loads.base import EquipmentCircuit
except ImportError:
    from loads.base import EquipmentCircuit

def get_bulb(rated_power_kw: float = 0.5) -> EquipmentCircuit:
    """
    Bulb: Industrial / Residential lighting load.
    """
    return EquipmentCircuit(
        equipment_type="bulb",
        rated_power_kw=rated_power_kw,
        rated_voltage_v=240.0,
        power_factor=0.98,
        opendss_params={
            "model": 1,
            "pf": 0.98
        },
        atp_params={
            "r_bulb": 115.0
        }
    )
