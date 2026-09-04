import pytest
import numpy as np
from src.transient.models import (
    TransformerSpec,
    TransformerWinding,
    ShortCircuitTest,
    ThreePhaseThevenin,
    TestBranch,
    SimulationConfig,
)
from src.transient.bctran_generator import BCTRANGenerator
from src.transient.atp_case_builder import ATPCaseBuilder
from src.transient.atp_runner import ATPRunner


@pytest.fixture
def sample_setup():
    tx_spec = TransformerSpec(
        name="test_tx",
        frequency_hz=50.0,
        windings=[
            TransformerWinding("HV", 33.0, 7.5, "Delta", 0.0),
            TransformerWinding("LV", 0.415, 7.5, "Wye", -30.0),
        ],
        short_circuit_tests=[
            ShortCircuitTest(1, 2, z_pos_pu=0.05, losses_pos_kw=10.0, z_zero_pu=0.05, losses_zero_kw=10.0)
        ],
        excitation_current_percent=0.5,
        excitation_loss_kw=25.0,
    )

    punch = BCTRANGenerator().generate(tx_spec)

    v_hv = np.array([6350 + 0j, -3175 - 5499j, -3175 + 5499j], dtype=complex)
    v_lv = np.array([240 + 0j, -120 - 207.8j, -120 + 207.8j], dtype=complex)

    z_hv = np.diag([0.01 + 0.05j, 0.01 + 0.05j, 0.01 + 0.05j])
    z_lv = np.diag([0.002 + 0.01j, 0.002 + 0.01j, 0.002 + 0.01j])

    up = ThreePhaseThevenin(v_th=v_hv, z_th=z_hv, v_pre=v_hv, i_pre=np.zeros(3, dtype=complex), frequency_hz=50.0)
    down = ThreePhaseThevenin(v_th=v_lv, z_th=z_lv, v_pre=v_lv, i_pre=np.zeros(3, dtype=complex), frequency_hz=50.0)

    test_branch = TestBranch(
        name="motor_1",
        branch_type="equipment",
        phases=(0, 1, 2),
        model={"R": 5.0, "L_mH": 10.0, "C_uF": 0.0},
        start_time_s=0.02,
        duration_s=0.1,
    )

    sim = SimulationConfig(t_start_s=0.0, t_stop_s=0.15, time_step_s=1e-4)

    return tx_spec, up, down, test_branch, sim, punch


def test_atp_case_builder_explicit(sample_setup, tmp_path):
    tx_spec, up, down, test_branch, sim, punch = sample_setup

    builder = ATPCaseBuilder()
    case_path = tmp_path / "test_explicit.ATP"

    content = builder.build_explicit(
        transformer=tx_spec,
        upstream=up,
        downstream=down,
        events=[test_branch],
        simulation=sim,
        bctran_punch=punch,
        output_path=str(case_path),
    )

    assert "14SRCA" in content
    assert "14VTHA" in content
    assert "$VINTAGE, 1," in content or "$VINTAGE,1," in content
    assert "/SWITCH" in content
    assert "/OUTPUT" in content

    # Test execution under Wine
    atp_res = ATPRunner().run(case_path)
    assert atp_res.return_code == 0
