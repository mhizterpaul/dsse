import pytest
import numpy as np
from src.simulation.runner import CoSimulationRunner
from src.transient.events import (
    SingleEquipmentSwitchEvent,
    SingleLineFaultEvent,
    EquipmentEquipmentCoEvent,
)


def test_co_simulation_runner_transients():
    runner = CoSimulationRunner()
    runner.initialize_plant_session(use_baseline_feeder=True, seed=42)

    ev1 = SingleEquipmentSwitchEvent(
        "ac_motor", start_time_s=0.02, duration_s=0.1, target="trans1", parameters={}
    )
    ev2 = SingleEquipmentSwitchEvent(
        "compressor", start_time_s=0.03, duration_s=0.1, target="trans1", parameters={}
    )
    co_ev = EquipmentEquipmentCoEvent(ev1, ev2)

    op = runner.run_steady_state_simulation(use_baseline_feeder=True, seed=42).operating_point

    t, v_dict, i_dict, meta = runner.measure_transients(
        op=op, event=co_ev, scenario_id="test_runner", feeder_idx=1, use_baseline_feeder=True
    )

    assert len(t) > 0
    assert "trans1_lv_boundary" in v_dict
    assert "trans1_lv_boundary" in i_dict
    assert v_dict["trans1_lv_boundary"].shape[1] == 3
    assert i_dict["trans1_lv_boundary"].shape[1] == 3
