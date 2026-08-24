from src.lv_networks.loads.base import EquipmentCircuit

def get_induction_plate(rated_power_kw: float = 3.5) -> EquipmentCircuit:
    """
    Induction Cooktop: Rectifier, DC link, high-frequency resonant inverter, resonant capacitor, induction coil R_eq + j*omega*L_eq.
    """
    return EquipmentCircuit(
        equipment_type="induction_plate",
        rated_power_kw=rated_power_kw,
        rated_voltage_v=240.0,
        power_factor=0.98,
        opendss_params={
            "model": 1,
            "pf": 0.98
        },
        atp_params={
            "c_resonant": 0.33e-6,
            "r_coil": 0.45,
            "l_coil": 45e-6,
            "resonant_freq_hz": 25000.0
        }
    )
