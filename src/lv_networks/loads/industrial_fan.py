from src.lv_networks.loads.base import EquipmentCircuit

def get_industrial_fan(rated_power_kw: float = 20.0) -> EquipmentCircuit:
    """
    Industrial Fan: Three-phase induction motor driving speed-squared aerodynamic fan load torque.
    """
    return EquipmentCircuit(
        equipment_type="industrial_fan",
        rated_power_kw=rated_power_kw,
        rated_voltage_v=415.0,
        power_factor=0.86,
        opendss_params={
            "model": 3,
            "pf": 0.86
        },
        atp_params={
            "r_stator": 0.03,
            "x_stator": 0.10,
            "x_magnetizing": 4.0,
            "r_rotor": 0.025,
            "x_rotor": 0.08,
            "fan_torque_coeff": 0.015
        }
    )
