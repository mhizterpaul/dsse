from src.lv_networks.loads.base import EquipmentCircuit

def get_dc_motor_inverter(rated_power_kw: float = 10.0) -> EquipmentCircuit:
    """
    DC Motor + PWM H-Bridge Inverter: Rectifier, DC link capacitor, PWM H-bridge, DC motor Ra, La, Back-EMF.
    """
    return EquipmentCircuit(
        equipment_type="dc_motor_inverter",
        rated_power_kw=rated_power_kw,
        rated_voltage_v=415.0,
        power_factor=0.92,
        opendss_params={
            "model": 1,
            "pf": 0.92
        },
        atp_params={
            "c_dc_link": 2200e-6,
            "r_armature": 0.12,
            "l_armature": 0.005,
            "k_back_emf": 0.85,
            "pwm_freq_hz": 5000.0
        }
    )
