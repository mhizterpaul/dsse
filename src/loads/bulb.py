try:
    from src.loads.base import EquipmentCircuit
except ImportError:
    from loads.base import EquipmentCircuit

def get_residential_bulb() -> EquipmentCircuit:
    """
    Residential Bulb: Low-power lighting load (0.1 kW).
    """
    return EquipmentCircuit(
        equipment_type="residential_bulb",
        rated_power_kw=0.1,
        rated_voltage_v=240.0,
        power_factor=0.98,
        opendss_params={
            "model": 1,
            "pf": 0.98
        },
        atp_params={
            "r_bulb": 576.0
        }
    )

def get_industrial_bulb() -> EquipmentCircuit:
    """
    Industrial Bulb: High-bay / flood lighting load (1.0 kW).
    """
    return EquipmentCircuit(
        equipment_type="industrial_bulb",
        rated_power_kw=1.0,
        rated_voltage_v=240.0,
        power_factor=0.95,
        opendss_params={
            "model": 1,
            "pf": 0.95
        },
        atp_params={
            "r_bulb": 57.6
        }
    )

def get_bulb() -> EquipmentCircuit:
    return get_residential_bulb()
