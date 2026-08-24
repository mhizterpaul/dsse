from dataclasses import dataclass, replace
from typing import Optional, Literal, Union

@dataclass
class SingleEquipmentSwitchEvent:
    equipment_type: str  # ac_motor, dc_motor_inverter, microwave, induction_plate, compressor, audio_amplifier, ups, industrial_fan
    start_time_s: float
    duration_s: float
    target: str
    parameters: dict

    @property
    def event_class(self) -> str:
        return "equipment_switch"

    @property
    def event_type(self) -> str:
        return self.equipment_type

@dataclass
class SingleLineFaultEvent:
    fault_type: str  # LG, LL, LLG, LLL, LC, LLC
    start_time_s: float
    duration_s: float
    target: str
    faulted_phases: tuple  # e.g., (0,), (0, 1), (0, 1, 2)
    fault_resistance: float
    parameters: dict

    @property
    def event_class(self) -> str:
        return "line_fault"

    @property
    def event_type(self) -> str:
        return self.fault_type

@dataclass
class EquipmentEquipmentCoEvent:
    event_1: SingleEquipmentSwitchEvent
    event_2: SingleEquipmentSwitchEvent

    @property
    def event_class(self) -> str:
        return "equipment_equipment_coevent"

    @property
    def event_type(self) -> str:
        return f"{self.event_1.event_type}_{self.event_2.event_type}"

    @property
    def is_simultaneous(self) -> bool:
        return self.event_1.start_time_s == self.event_2.start_time_s

    @property
    def time_offset_s(self) -> float:
        return abs(self.event_2.start_time_s - self.event_1.start_time_s)

    def with_time_shift(self, offset_s: float):
        ev2_shifted = replace(self.event_2, start_time_s=self.event_1.start_time_s + offset_s)
        return EquipmentEquipmentCoEvent(event_1=self.event_1, event_2=ev2_shifted)

@dataclass
class LineFaultLineFaultCoEvent:
    event_1: SingleLineFaultEvent
    event_2: SingleLineFaultEvent

    @property
    def event_class(self) -> str:
        return "line_fault_line_fault_coevent"

    @property
    def event_type(self) -> str:
        return f"{self.event_1.event_type}_{self.event_2.event_type}"

    @property
    def is_simultaneous(self) -> bool:
        return self.event_1.start_time_s == self.event_2.start_time_s

    @property
    def time_offset_s(self) -> float:
        return abs(self.event_2.start_time_s - self.event_1.start_time_s)

    def with_time_shift(self, offset_s: float):
        ev2_shifted = replace(self.event_2, start_time_s=self.event_1.start_time_s + offset_s)
        return LineFaultLineFaultCoEvent(event_1=self.event_1, event_2=ev2_shifted)

@dataclass
class EquipmentLineFaultCoEvent:
    event_1: SingleEquipmentSwitchEvent
    event_2: SingleLineFaultEvent

    @property
    def event_class(self) -> str:
        return "equipment_line_fault_coevent"

    @property
    def event_type(self) -> str:
        return f"{self.event_1.event_type}_{self.event_2.event_type}"

    @property
    def is_simultaneous(self) -> bool:
        return self.event_1.start_time_s == self.event_2.start_time_s

    @property
    def time_offset_s(self) -> float:
        return abs(self.event_2.start_time_s - self.event_1.start_time_s)

    def with_time_shift(self, offset_s: float):
        ev2_shifted = replace(self.event_2, start_time_s=self.event_1.start_time_s + offset_s)
        return EquipmentLineFaultCoEvent(event_1=self.event_1, event_2=ev2_shifted)

TransientEvent = Union[
    SingleEquipmentSwitchEvent,
    SingleLineFaultEvent,
    EquipmentEquipmentCoEvent,
    LineFaultLineFaultCoEvent,
    EquipmentLineFaultCoEvent
]
