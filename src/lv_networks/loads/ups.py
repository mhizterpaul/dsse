from src.lv_networks.loads.base import EquipmentCircuit

def get_ups(rated_power_kw: float = 8.0) -> EquipmentCircuit:
    """
    Uninterruptible Power Supply (UPS): Battery bank, DC-link, bidirectional inverter, AC-side filter interface.
    """
    return EquipmentCircuit(
        equipment_type="ups",
        rated_power_kw=rated_power_kw,
        rated_voltage_v=415.0,
        power_factor=1.0,
        opendss_params={
            "model": 1,
            "pf": 1.0,
            "is_storage": True
        },
        atp_params={
            "v_dc_battery": 380.0,
            "r_internal": 0.08,
            "c_dc_link": 3300e-6,
            "l_ac_filter": 1.2e-3
        }
    )
