try:
    from src.loads.base import EquipmentCircuit
except ImportError:
    from loads.base import EquipmentCircuit

def get_residential_fan() -> EquipmentCircuit:
    """
    Residential Fan: Ceiling / standing fan (0.1 kW).
    """
    return EquipmentCircuit(
        equipment_type="residential_fan",
        rated_power_kw=0.1,
        rated_voltage_v=240.0,
        power_factor=0.90,
        opendss_params={
            "model": 1,
            "pf": 0.90
        },
        atp_params={
            "r_fan": 500.0,
            "l_fan": 0.1
        }
    )

def get_fan() -> EquipmentCircuit:
    return get_residential_fan()
