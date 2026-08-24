from src.lv_networks.loads.base import EquipmentCircuit

def get_microwave(rated_power_kw: float = 1.8) -> EquipmentCircuit:
    """
    Microwave Oven: Input rectifier, PFC, DC-link capacitor, HV transformer, diode voltage doubler, magnetron.
    """
    return EquipmentCircuit(
        equipment_type="microwave",
        rated_power_kw=rated_power_kw,
        rated_voltage_v=240.0,
        power_factor=0.95,
        opendss_params={
            "model": 1,
            "pf": 0.95
        },
        atp_params={
            "c_dc_link": 470e-6,
            "hv_ratio": 10.0,
            "c_doubler": 1.0e-6,
            "r_magnetron": 250.0
        }
    )
