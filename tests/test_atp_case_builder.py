import os
import pytest
import numpy as np
from pathlib import Path

from src.transient.atp_case_builder import (
    ATPCaseBuilder,
    ATPCaseValidator,
    TransformerSpec,
    TransformerWinding,
    ShortCircuitTest,
    SourceModel,
    ThreePhaseState,
    LineModel,
    LoadModel,
    TransientEvent,
    SimulationConfig,
    atp_e8,
    atp_misc_card,
    fmt_type14_source,
    fmt_branch,
    fmt_switch,
)
from src.transient.atp_runner import ATPRunner
from src.transient.atp_parser import ATPOutputReader
from src.transient.events import (
    SingleEquipmentSwitchEvent,
    SingleLineFaultEvent,
    EquipmentLineFaultCoEvent,
    EquipmentEquipmentCoEvent,
)


def test_atp_e8_field_width():
    s1 = atp_e8(1e-4)
    assert len(s1) == 8
    assert float(s1) == pytest.approx(1e-4)

    s2 = atp_e8(0.15)
    assert len(s2) == 8
    assert float(s2) == pytest.approx(0.15)

    s3 = atp_e8(0.0)
    assert len(s3) == 8
    assert float(s3) == pytest.approx(0.0)


def test_atp_misc_card_format():
    card = atp_misc_card(dt=1e-4, tmax=0.15, xopt=0.0, copt=0.0)
    assert len(card) == 56  # 7 fields * 8 chars
    # Check field 1
    assert card[0:8] == "1.00E-04"
    # Check field 2
    assert card[8:16] == "1.50E-01"
    # Check field 3
    assert card[16:24] == "0.00E+00"


def test_fmt_type14_source_format():
    src_card = fmt_type14_source("SRCA", amplitude=337.997, frequency=50.0, phase_deg=-60.0, t_start=-1.0, t_stop=1000.0)
    assert src_card.startswith("14SRCA  ")
    assert len(src_card) <= 80
    # No artificial '0' field
    assert "SRCA       0" not in src_card


def test_fmt_branch_and_switch():
    b_card = fmt_branch("SEC_A", "L0A", 10.0, 1.5, 0.0)
    assert len(b_card) <= 80
    assert b_card.startswith("  SEC_A L0A   ")

    sw_card = fmt_switch("SEC_A", "F0_A", 0.05, 0.10)
    assert len(sw_card) <= 80
    assert sw_card.startswith("  SEC_A F0_A  ")


def test_atp_case_validator():
    valid_deck = (
        "BEGIN NEW DATA CASE\n"
        "C  Test Case\n"
        "POWER FREQUENCY                      50.\n"
        "$DUMMY, XYZ000\n"
        "C  dT  >< Tmax >< Xopt >< Copt ><Epsiln>\n"
        "1.00E-041.50E-010.00E+000.00E+000.00E+000.00E+000.00E+00\n"
        "    1000       1       1       1       1       0       0       1       0\n"
        "/BRANCH\n"
        "  SRCA  MB_A                  0.1000    0.3183    0.0000\n"
        "/SWITCH\n"
        "  SECA  L0A      -1.0000  100.0000\n"
        "/SOURCE\n"
        "14SRCA     337.997    50.000     0.000    -1.000  1000.000\n"
        "/OUTPUT\n"
        "  SECA  SECB  SECC\n"
        "BLANK BRANCH\n"
        "BLANK SWITCH\n"
        "BLANK SOURCE\n"
        "BLANK OUTPUT\n"
        "BLANK PLOT\n"
        "BEGIN NEW DATA CASE\n"
        "BLANK"
    )
    ATPCaseValidator.validate_content(valid_deck)

    invalid_deck = valid_deck + "\n" + "X" * 85
    with pytest.raises(ValueError, match="exceeds 80 characters"):
        ATPCaseValidator.validate_content(invalid_deck)


def test_build_explicit_and_run_atp(tmp_path):
    tx = TransformerSpec(
        name="Tx1",
        frequency_hz=50.0,
        windings=[
            TransformerWinding("HV", 11.0, 7.5, "Y", 0.0),
            TransformerWinding("LV", 0.415, 7.5, "Y", 0.0),
        ],
        short_circuit_tests=[
            ShortCircuitTest(1, 2, z_pos_pu=0.05, losses_pos_kw=10.0, z_zero_pu=0.04, losses_zero_kw=8.0)
        ],
        excitation_current_percent=1.0,
        excitation_loss_kw=5.0,
    )

    src = SourceModel("GRID", 50.0, ThreePhaseState((239.0, 239.0, 239.0), (0.0, -120.0, 120.0)))
    line = LineModel("Line1", "main_bus", "feeder1_head", 1.0, 0.1, 0.1, 1e-9)
    loads = [LoadModel("L0", "sec", 10.0, 2.0, 10.0, 0.001)]

    ev1 = SingleEquipmentSwitchEvent("ac_motor", 0.05, 0.05, "trans1", {})
    ev2 = SingleLineFaultEvent("ag", 0.08, 0.05, "trans1", (0,), 0.001, {})
    co_ev = EquipmentLineFaultCoEvent(ev1, ev2)

    sim = SimulationConfig(0.0, 0.15, 1e-4)

    atp_path = tmp_path / "test_case.ATP"
    builder = ATPCaseBuilder()
    content = builder.build_explicit(tx, src, line, loads, co_ev, sim, output_path=str(atp_path), scenario_id="pytest_scenario")

    assert atp_path.exists()
    ATPCaseValidator.validate_file(atp_path)

    res = ATPRunner().run(atp_path)
    assert res.return_code == 0

    emt = ATPOutputReader().read(res, co_ev, transformer_id="trans1")
    assert emt is not None
    assert len(emt.time_s) > 0
    assert "trans1" in emt.voltages
    assert emt.voltages["trans1"].shape[1] == 3
