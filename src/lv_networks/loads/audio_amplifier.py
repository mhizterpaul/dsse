from src.lv_networks.loads.base import EquipmentCircuit

def get_audio_amplifier(rated_power_kw: float = 2.5) -> EquipmentCircuit:
    """
    Audio Amplifier: AC supply, rectifier, DC-link supply capacitors, Class-D H-bridge, LC output filter, speaker load.
    """
    return EquipmentCircuit(
        equipment_type="audio_amplifier",
        rated_power_kw=rated_power_kw,
        rated_voltage_v=240.0,
        power_factor=0.90,
        opendss_params={
            "model": 1,
            "pf": 0.90
        },
        atp_params={
            "c_supply_bank": 10000e-6,
            "l_filter": 22e-6,
            "c_filter": 0.47e-6,
            "r_speaker": 4.0
        }
    )
