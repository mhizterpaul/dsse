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
        p_kw = float(eq_model.rated_power_kw)
        v_v = float(eq_model.rated_voltage_v)
        pf = float(eq_model.power_factor)

        if p_kw <= 0 or v_v <= 0 or pf <= 0 or pf > 1.0:
            raise ValueError(
                f"Invalid rating parameters for '{self.equipment_type}': P={p_kw}, V={v_v}, pf={pf}"
            )

        z_mag = (v_v ** 2) / ((p_kw * 1000.0) / pf)
        r_val = z_mag * pf
        x_val = z_mag * np.sqrt(max(0.0, 1.0 - pf ** 2))
        l_mH_val = (x_val / (2.0 * np.pi * frequency_hz)) * 1000.0

        model_dict = {
            "R": r_val,
            "L_mH": l_mH_val,
            "C_uF": 0.0,
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
class MultiEquipmentCoEvent:
    events: List[SingleEquipmentSwitchEvent]

    @property
    def event_class(self) -> str:
        return "multi_equipment_coevent"

    @property
    def event_type(self) -> str:
        return f"multi_equipment_{len(self.events)}"

    @property
    def start_time_s(self) -> float:
        return self.events[0].start_time_s if self.events else 0.02

    @property
    def duration_s(self) -> float:
        return self.events[0].duration_s if self.events else 0.10

    def to_test_branches(self, frequency_hz: float) -> List[TestBranch]:
        branches = []
        for idx, ev in enumerate(self.events):
            b_list = ev.to_test_branches(frequency_hz)
            for b in b_list:
                branches.append(
                    TestBranch(
                        name=f"{b.name}_{idx}",
                        branch_type=b.branch_type,
                        phases=b.phases,
                        model=b.model,
                        start_time_s=b.start_time_s,
                        duration_s=b.duration_s,
                    )
                )
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


@dataclass
class NoLoadEvent:
    start_time_s: float = 0.02
    duration_s: float = 0.10

    @property
    def event_class(self) -> str:
        return "no_load"

    @property
    def event_type(self) -> str:
        return "no_load"

    def to_test_branches(self, frequency_hz: float) -> List[TestBranch]:
        return []


TransientEvent = Union[
    SingleEquipmentSwitchEvent,
    SingleLineFaultEvent,
    EquipmentEquipmentCoEvent,
    EquipmentLineFaultCoEvent,
    NoLoadEvent,
]
