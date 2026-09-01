import sys
import json
import unittest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner, extract_fault_info
from src.transient.events import (
    SingleEquipmentSwitchEvent,
    SingleLineFaultEvent,
    EquipmentEquipmentCoEvent,
    EquipmentLineFaultCoEvent
)


class TestEventTransients(unittest.TestCase):

    def test_extract_fault_info_load_fault_event(self):
        """
        Tests extract_fault_info extracts fault parameters from EquipmentLineFaultCoEvent.
        """
        ev1 = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, "feeder1_head", {})
        ev2 = SingleLineFaultEvent("LG", 0.02, 0.04, "feeder1_head", (0,), 0.05, {"config_id": "AG"})
        co_ev = EquipmentLineFaultCoEvent(ev1, ev2)

        fault_json = extract_fault_info(co_ev)
        self.assertNotEqual(fault_json, "", "extract_fault_info returned empty string for load-fault event")

        fault_data = json.loads(fault_json)
        self.assertEqual(fault_data["fault_type"], "LG")
        self.assertEqual(fault_data["fault_resistance_ohm"], 0.05)
        self.assertEqual(fault_data["faulted_phases"], [0])
        self.assertEqual(fault_data["config_id"], "AG")

        print("LOGGED FAULT INFO:", fault_data)

    def test_extract_fault_info_load_load_event(self):
        """
        Tests extract_fault_info returns empty JSON string for pure load-load co-events when no fault present.
        """
        ev1 = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, "feeder1_head", {})
        ev2 = SingleEquipmentSwitchEvent("dc_motor_inverter", 0.02, 0.04, "feeder1_head", {})
        co_ev = EquipmentEquipmentCoEvent(ev1, ev2)

        fault_json = extract_fault_info(co_ev)
        self.assertEqual(fault_json, "", "extract_fault_info should return empty string for load-load events")

    def test_retrieve_event_and_transient_info(self):
        """
        Tests CoSimulationRunner to initialize plant session, execute simulation for a co-event,
        retrieve simulation result object, and log event info and transient waveforms.
        """
        runner = CoSimulationRunner()
        runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

        ev1 = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, "feeder1_head", {})
        ev2 = SingleLineFaultEvent("LLG", 0.02, 0.04, "feeder1_head", (0, 1), 0.05, {"config_id": "ABG"})
        co_ev = EquipmentLineFaultCoEvent(ev1, ev2)

        # Extract & log fault info
        fault_info_str = extract_fault_info(co_ev)
        print("LOGGED FAULT INFO:", fault_info_str)
        self.assertNotEqual(fault_info_str, "")

        # Run co-event simulation
        sim_res = runner.run_simulation(
            events=[co_ev],
            use_baseline_transformers=True,
            include_load_event=True,
            include_fault_event=True,
            scenario_id="test_transient_logging",
            seed=42,
            reinitialize_plant=False
        )

        self.assertIsNotNone(sim_res)
        self.assertIsInstance(sim_res.processed_consumer_units, dict)

        # Log transient waveform information
        m_id = "trans1_lv_boundary_consumer_unit"
        unit_data = sim_res.processed_consumer_units.get(m_id, {})

        v_raw = unit_data.get("raw_voltage")
        i_raw = unit_data.get("raw_current")

        print(f"TRANSIENT WAVEFORMS for {m_id}:")
        print(f"  Time points: {len(sim_res.time_s)}")
        print(f"  Voltage shape: {v_raw.shape if v_raw is not None else None}")
        print(f"  Current shape: {i_raw.shape if i_raw is not None else None}")


if __name__ == "__main__":
    unittest.main()
