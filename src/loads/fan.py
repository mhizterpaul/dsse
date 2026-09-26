try:
    from src.loads.base import EquipmentCircuit
except ImportError:
    from loads.base import EquipmentCircuit

def get_fan(rated_power_kw: float = 0.8) -> EquipmentCircuit:
    """
    Fan: Residential / Commercial ceiling and cooling fan.
    """
    return EquipmentCircuit(
        equipment_type="fan",
        rated_power_kw=rated_power_kw,
        rated_voltage_v=240.0,
        power_factor=0.90,
        opendss_params={
            "model": 1,
            "pf": 0.90
        },
        atp_params={
            "r_fan": 180.0,
            "l_fan": 0.05
        }
    )
