from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EquipmentCircuit:
    equipment_type: str
    rated_power_kw: float
    rated_voltage_v: float
    power_factor: float
    opendss_params: Dict[str, Any]
    atp_params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equipment_type": self.equipment_type,
            "rated_power_kw": self.rated_power_kw,
            "rated_voltage_v": self.rated_voltage_v,
            "power_factor": self.power_factor,
            "opendss_params": self.opendss_params,
            "atp_params": self.atp_params
        }
