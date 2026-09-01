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

LOG_FILE = PROJECT_ROOT / "test_transient_logging.log"


def log_test_entry(test_name: str, status: str, details: dict):
    """
    Appends test execution details, status (SUCCESSFUL/FAILED), fault info, and transient metrics to log file.
    """
    log_line = f"=== Test: {test_name} | Status: {status} ===\n"
    for k, v in details.items():
        log_line += f"  {k}: {v}\n"
    log_line += "\n"
    with open(LOG_FILE, "a") as f:
        f.write(log_line)


class TestEventTransients(unittest.TestCase):

    def test_extract_fault_info_load_fault_event(self):
        """
        Tests extract_fault_info extracts fault parameters from EquipmentLineFaultCoEvent.
        """
        test_name = "test_extract_fault_info_load_fault_event"
        try:
            ev1 = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, "feeder1_head", {})
            ev2 = SingleLineFaultEvent("LG", 0.02, 0.04, "feeder1_head", (0,), 0.05, {"config_id": "AG"})
            co_ev = EquipmentLineFaultCoEvent(ev1, ev2)

            fault_json = extract_fault_info(co_ev)
            self.assertNotEqual(fault_json, "", "extract_fault_info returned empty string for load-fault event")

            fault_data = json.loads(fault_json)
            self.assertEqual(fault_data["fault_type"], "LG")
            self.assertEqual(fault_data["fault_resistance_ohm"], 0.05)
            self.assertEqual(fault_data["faulted_phases"], [0])

            details = {
                "fault_info_extracted": fault_data,
                "message": "Fault info successfully extracted from event and verified."
            }
            log_test_entry(test_name, "SUCCESSFUL", details)
            print(f"[{test_name}] SUCCESSFUL")
        except Exception as e:
            log_test_entry(test_name, "FAILED", {"error": str(e)})
            print(f"[{test_name}] FAILED: {e}")
            raise

    def test_extract_fault_info_load_load_event(self):
        """
        Tests extract_fault_info returns empty JSON string for pure load-load co-events when no fault present.
        """
        test_name = "test_extract_fault_info_load_load_event"
        try:
            ev1 = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, "feeder1_head", {})
            ev2 = SingleEquipmentSwitchEvent("dc_motor_inverter", 0.02, 0.04, "feeder1_head", {})
            co_ev = EquipmentEquipmentCoEvent(ev1, ev2)

            fault_json = extract_fault_info(co_ev)
            self.assertEqual(fault_json, "", "extract_fault_info should return empty string for load-load events")

            details = {
                "fault_info_extracted": "None (pure load-load event)",
                "message": "Correctly handled non-fault event."
            }
            log_test_entry(test_name, "SUCCESSFUL", details)
            print(f"[{test_name}] SUCCESSFUL")
        except Exception as e:
            log_test_entry(test_name, "FAILED", {"error": str(e)})
            print(f"[{test_name}] FAILED: {e}")
            raise

    def test_retrieve_event_and_transient_info(self):
        """
        Tests CoSimulationRunner to initialize plant session in OpenDSS, execute simulation for a co-event,
        retrieve simulation result object, and log fault info from OpenDSS and transients from ATP-EMTP.
        """
        test_name = "test_retrieve_event_and_transient_info"
        try:
            runner = CoSimulationRunner()
            runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

            ev1 = SingleEquipmentSwitchEvent("ac_motor", 0.02, 0.04, "feeder1_head", {})
            ev2 = SingleLineFaultEvent("LLG", 0.02, 0.04, "feeder1_head", (0, 1), 0.05, {"config_id": "ABG"})
            co_ev = EquipmentLineFaultCoEvent(ev1, ev2)

            # Extract fault info
            fault_info_str = extract_fault_info(co_ev)
            self.assertNotEqual(fault_info_str, "")
            fault_data = json.loads(fault_info_str)

            # Run co-event simulation (OpenDSS power flow + ATP-EMTP transients)
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

            # Retrieve transient metrics from ATP-EMTP output
            m_id = "trans1_lv_boundary_consumer_unit"
            unit_data = sim_res.processed_consumer_units.get(m_id, {})

            v_raw = unit_data.get("raw_voltage")
            i_raw = unit_data.get("raw_current")

            details = {
                "fault_info_from_opendss": fault_data,
                "atp_transient_boundary_unit": m_id,
                "time_series_points": len(sim_res.time_s),
                "voltage_shape": str(v_raw.shape) if v_raw is not None else "None",
                "current_shape": str(i_raw.shape) if i_raw is not None else "None",
                "max_voltage_v": float(np.abs(v_raw).max()) if v_raw is not None else 0.0,
                "max_current_a": float(np.abs(i_raw).max()) if i_raw is not None else 0.0,
                "steady_state_measurements_count": len(sim_res.steady_state_measurements)
            }

            log_test_entry(test_name, "SUCCESSFUL", details)
            print(f"[{test_name}] SUCCESSFUL")
        except Exception as e:
            log_test_entry(test_name, "FAILED", {"error": str(e)})
            print(f"[{test_name}] FAILED: {e}")
            raise


if __name__ == "__main__":
    unittest.main()
