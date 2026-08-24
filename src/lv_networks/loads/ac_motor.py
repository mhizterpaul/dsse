from src.lv_networks.loads.base import EquipmentCircuit

def get_ac_motor(rated_power_kw: float = 15.0) -> EquipmentCircuit:
    """
    3-Phase Induction Motor: Stator R, L, magnetizing branch, rotor R, L, mechanical inertia.
    """
    return EquipmentCircuit(
        equipment_type="ac_motor",
        rated_power_kw=rated_power_kw,
        rated_voltage_v=415.0,
        power_factor=0.85,
        opendss_params={
            "model": 3,
            "pf": 0.85,
            "h": 1.2
        },
        atp_params={
            "r_stator": 0.05,
            "x_stator": 0.15,
            "x_magnetizing": 3.5,
            "r_rotor": 0.04,
            "x_rotor": 0.12,
            "inertia_j": 0.8
        }
    )
