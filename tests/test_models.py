import pytest
import numpy as np
from src.transient.models import (
    ThreePhasePhasor,
    ThreePhaseThevenin,
    TestBranch,
    TransformerSpec,
    TransformerWinding,
    ShortCircuitTest,
    SimulationConfig,
)


def test_three_phase_phasor():
    phasor = ThreePhasePhasor((100 + 0j, -50 - 86.6025j, -50 + 86.6025j))
    mags = phasor.magnitudes
    assert len(mags) == 3
    assert pytest.approx(mags[0], abs=1e-2) == 100.0
    assert pytest.approx(mags[1], abs=1e-2) == 100.0
    assert pytest.approx(mags[2], abs=1e-2) == 100.0


def test_three_phase_thevenin_reconstruction():
    v_pre = np.array([230 + 0j, -115 - 199.185j, -115 + 199.185j], dtype=complex)
    i_pre = np.array([10 - 5j, -5 + 8j, -5 - 3j], dtype=complex)
    z_th = np.diag([0.01 + 0.05j, 0.01 + 0.05j, 0.01 + 0.05j])

    v_th = v_pre + (z_th @ i_pre)

    thevenin = ThreePhaseThevenin(
        v_th=v_th, z_th=z_th, v_pre=v_pre, i_pre=i_pre, frequency_hz=50.0
    )

    v_recon = thevenin.reconstruct_v_pre()
    np.testing.assert_allclose(v_recon, v_pre, atol=1e-5)
    assert thevenin.validate_equivalence()


def test_test_branch_dataclass():
    tb = TestBranch(
        name="motor_01",
        branch_type="equipment",
        phases=(0, 1, 2),
        model={"R": 5.0, "L_mH": 10.0, "C_uF": 0.0},
        start_time_s=0.02,
        duration_s=1.5,
    )
    assert tb.end_time_s == 1.52
    assert tb.branch_type == "equipment"
