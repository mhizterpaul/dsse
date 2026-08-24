from src.lv_networks.loads.base import EquipmentCircuit

def get_compressor(rated_power_kw: float = 4.0) -> EquipmentCircuit:
    """
    Compressor: Single-phase AC induction motor driving reciprocating/scroll compressor load torque.
    """
    return EquipmentCircuit(
        equipment_type="compressor",
        rated_power_kw=rated_power_kw,
        rated_voltage_v=240.0,
        power_factor=0.82,
        opendss_params={
            "model": 3,
            "pf": 0.82
        },
        atp_params={
            "r_stator": 0.15,
            "x_stator": 0.25,
            "r_rotor": 0.10,
            "x_rotor": 0.20,
            "torque_load_constant": 12.5
        }
    )
