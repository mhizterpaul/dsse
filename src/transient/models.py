from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal, Any
import numpy as np


@dataclass(frozen=True)
class ThreePhasePhasor:
    values: Tuple[complex, complex, complex]

    @property
    def magnitudes(self) -> Tuple[float, float, float]:
        return (abs(self.values[0]), abs(self.values[1]), abs(self.values[2]))

    @property
    def angles_deg(self) -> Tuple[float, float, float]:
        return (
            float(np.rad2deg(np.angle(self.values[0]))),
            float(np.rad2deg(np.angle(self.values[1]))),
            float(np.rad2deg(np.angle(self.values[2]))),
        )


@dataclass(frozen=True)
class ThreePhaseThevenin:
    v_th: np.ndarray  # shape (3,), complex RMS volts
    z_th: np.ndarray  # shape (3, 3), complex ohms
    v_pre: np.ndarray  # shape (3,), complex RMS volts
    i_pre: np.ndarray  # shape (3,), complex RMS amperes
    frequency_hz: float

    @property
    def y_th(self) -> np.ndarray:
        return np.linalg.inv(self.z_th)

    @property
    def i_n(self) -> np.ndarray:
        return self.y_th @ self.v_th

    def reconstruct_v_pre(self) -> np.ndarray:
        """
        Reconstructs pre-event terminal voltage using sign convention:
        V_reconstructed = V_th - Z_th @ I_pre
        """
        return self.v_th - (self.z_th @ self.i_pre)

    def validate_equivalence(self, atol_v: float = 1e-2, rtol_v: float = 5e-3) -> bool:
        """
        Validates that reconstructed pre-event voltage matches V_pre.
        """
        v_recon = self.reconstruct_v_pre()
        diff = np.linalg.norm(v_recon - self.v_pre)
        norm_v = np.linalg.norm(self.v_pre)
        rel_err = diff / max(norm_v, 1e-6)
        if rel_err > rtol_v and diff > atol_v:
            raise ValueError(
                f"Thévenin reconstruction error exceeds tolerance: rel_err={rel_err:.6f}, diff={diff:.6f} V"
            )
        return True


@dataclass(frozen=True)
class TwoPortEquivalent:
    z11_ohm: np.ndarray  # shape (3, 3), complex ohms
    z12_ohm: np.ndarray  # shape (3, 3), complex ohms
    z21_ohm: np.ndarray  # shape (3, 3), complex ohms
    z22_ohm: np.ndarray  # shape (3, 3), complex ohms
    v1_th: np.ndarray  # shape (3,), complex RMS volts
    v2_th: np.ndarray  # shape (3,), complex RMS volts
    frequency_hz: float


@dataclass(frozen=True)
class TestBranch:
    name: str
    branch_type: str  # "load", "line", "fault", "equipment"
    phases: Tuple[int, ...]  # e.g., (0, 1, 2)
    model: dict  # dict with keys like R, L_mH, C_uF, fault_resistance_ohm
    start_time_s: float
    duration_s: float

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s


@dataclass(frozen=True)
class TransformerWinding:
    name: str
    rated_kv: float  # line-to-line rated kV
    rated_mva: float  # rated 3-phase MVA
    connection: str  # "Delta", "Wye", "WyeG"
    phase_shift_deg: float  # vector group phase shift angle (e.g. 0.0, -30.0)


@dataclass(frozen=True)
class ShortCircuitTest:
    winding_i: int
    winding_j: int
    z_pos_pu: float
    losses_pos_kw: float
    z_zero_pu: Optional[float] = None
    losses_zero_kw: Optional[float] = None


@dataclass(frozen=True)
class TransformerSpec:
    name: str
    frequency_hz: float
    windings: List[TransformerWinding]
    short_circuit_tests: List[ShortCircuitTest]
    excitation_current_percent: float
    excitation_loss_kw: float


@dataclass(frozen=True)
class SimulationConfig:
    t_start_s: float
    t_stop_s: float
    time_step_s: float
