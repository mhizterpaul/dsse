from dataclasses import dataclass, replace
from typing import Optional, Literal, Union, List, Tuple
import numpy as np

from src.transient.models import TestBranch


@dataclass
class SingleEquipmentSwitchEvent:
    equipment_type: str  # ac_motor, dc_motor_inverter, microwave, induction_plate, compressor, audio_amplifier, ups, industrial_fan
    start_time_s: float
    duration_s: float
    target: str  # equipment ID or target transformer
    parameters: dict

    @property
    def event_class(self) -> str:
        return "equipment_switch"

    @property
    def event_type(self) -> str:
        return self.equipment_type

    def to_test_branches(self, frequency_hz: float) -> List[TestBranch]:
        from src.loads import get_equipment_model

        eq_model = get_equipment_model(self.equipment_type)
        r_val = None
        for key in [
            "r_stator",
            "r_armature",
            "r_coil",
            "r_internal",
            "r_magnetron",
            "r_speaker",
        ]:
            if key in eq_model.atp_params:
                r_val = float(eq_model.atp_params[key])
                break

        if r_val is None:
            raise ValueError(
                f"Equipment model '{self.equipment_type}' missing required R in atp_params"
            )

        l_mH_val = 0.0
        for key in ["l_armature", "l_coil", "l_ac_filter", "l_filter"]:
            if key in eq_model.atp_params:
                l_mH_val = float(eq_model.atp_params[key]) * 1000.0
                break

        if l_mH_val == 0.0:
            for key in ["x_stator", "x_rotor"]:
                if key in eq_model.atp_params:
                    x_val = float(eq_model.atp_params[key])
                    l_mH_val = (x_val / (2.0 * np.pi * frequency_hz)) * 1000.0
                    break

        c_uF_val = 0.0
        for key in [
            "c_doubler",
            "c_resonant",
            "c_dc_link",
            "c_supply_bank",
            "c_filter",
        ]:
            if key in eq_model.atp_params:
                c_uF_val = float(eq_model.atp_params[key]) * 1e6
                break

        model_dict = {
            "R": r_val,
            "L_mH": l_mH_val,
            "C_uF": c_uF_val,
            "equipment_type": self.equipment_type,
        }

        return [
            TestBranch(
                name=f"eq_{self.equipment_type}_{self.target}",
                branch_type="equipment",
                phases=(0, 1, 2),
                model=model_dict,
                start_time_s=self.start_time_s,
                duration_s=self.duration_s,
            )
        ]


@dataclass
class SingleLineFaultEvent:
    fault_type: str  # LG, LL, LLG, LLL, LC, LLC
    start_time_s: float
    duration_s: float
    target: str  # target line or bus
    faulted_phases: tuple  # e.g., (0,), (0, 1), (0, 1, 2)
    fault_resistance: float
    parameters: dict

    @property
    def event_class(self) -> str:
        return "line_fault"

    @property
    def event_type(self) -> str:
        return self.fault_type

    def to_test_branches(self, frequency_hz: float) -> List[TestBranch]:
        model_dict = {
            "fault_type": self.fault_type,
            "fault_resistance_ohm": float(self.fault_resistance),
            "faulted_phases": self.faulted_phases,
        }
        return [
            TestBranch(
                name=f"fault_{self.fault_type}_{self.target}",
                branch_type="fault",
                phases=tuple(self.faulted_phases),
                model=model_dict,
                start_time_s=self.start_time_s,
                duration_s=self.duration_s,
            )
        ]


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
        ev2_shifted = replace(
            self.event_2, start_time_s=self.event_1.start_time_s + offset_s
        )
        return EquipmentEquipmentCoEvent(event_1=self.event_1, event_2=ev2_shifted)

    def to_test_branches(self, frequency_hz: float) -> List[TestBranch]:
        branches = []
        branches.extend(self.event_1.to_test_branches(frequency_hz))
        branches.extend(self.event_2.to_test_branches(frequency_hz))
        return branches


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
        ev2_shifted = replace(
            self.event_2, start_time_s=self.event_1.start_time_s + offset_s
        )
        return EquipmentLineFaultCoEvent(event_1=self.event_1, event_2=ev2_shifted)

    def to_test_branches(self, frequency_hz: float) -> List[TestBranch]:
        branches = []
        branches.extend(self.event_1.to_test_branches(frequency_hz))
        branches.extend(self.event_2.to_test_branches(frequency_hz))
        return branches


TransientEvent = Union[
    SingleEquipmentSwitchEvent,
    SingleLineFaultEvent,
    EquipmentEquipmentCoEvent,
    EquipmentLineFaultCoEvent,
]
