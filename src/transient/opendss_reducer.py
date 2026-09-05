import numpy as np
import traceback
from typing import Dict, Any, List, Optional, Tuple
import src.power_plant.plant as plant
from src.transient.models import ThreePhaseThevenin


class OpenDSSReducer:
    """
    OpenDSS Network Reducer that extracts multi-phase Thévenin equivalents (V_th, Z_th)
    at transformer HV ports and downstream LV test buses.
    Ensures that test equipment and faults are strictly EXCLUDED from OpenDSS pre-event steady-state
    and network reduction, leaving ATP as the sole EMT model of test branches.
    """

    def __init__(self):
        pass

    def _extract_bus_complex_voltages(self, dss_instance: Any, bus_name: str) -> np.ndarray:
        if not dss_instance.Circuit.SetActiveBus(bus_name):
            raise ValueError(f"Bus '{bus_name}' could not be activated in OpenDSS")
        v_raw = dss_instance.Bus.Voltages()
        if len(v_raw) < 6:
            raise ValueError(
                f"Bus '{bus_name}' voltages returned fewer than 6 values: {v_raw}"
            )
        return np.array(
            [
                complex(v_raw[0], v_raw[1]),
                complex(v_raw[2], v_raw[3]),
                complex(v_raw[4], v_raw[5]),
            ],
            dtype=complex,
        )

    def _extract_bus_zsc_matrix(self, dss_instance: Any, bus_name: str) -> np.ndarray:
        if not dss_instance.Circuit.SetActiveBus(bus_name):
            raise ValueError(f"Bus '{bus_name}' could not be activated in OpenDSS")
        dss_instance.Text.Command("solve mode=faultstudy")
        if not dss_instance.Circuit.SetActiveBus(bus_name):
            raise ValueError(
                f"Bus '{bus_name}' could not be reactivated in OpenDSS faultstudy mode"
            )
        raw_z = dss_instance.Bus.ZscMatrix()
        if not raw_z or len(raw_z) < 18:
            raise ValueError(
                f"Bus '{bus_name}' ZscMatrix returned insufficient data: {raw_z}"
            )
        z_mat = np.zeros((3, 3), dtype=complex)
        idx = 0
        for i in range(3):
            for j in range(3):
                z_mat[i, j] = complex(raw_z[idx], raw_z[idx + 1])
                idx += 2
        return z_mat

    def reduce_upstream_thevenin(
        self, dss_instance: Any, tx_name: str = "trans1"
    ) -> ThreePhaseThevenin:
        """
        Calculates 3-phase Thévenin equivalent (V_th_HV, Z_th_HV) looking upstream of the transformer HV bus.
        """
        tx_elem = f"Transformer.{tx_name}"
        if not dss_instance.Circuit.SetActiveElement(tx_elem):
            raise ValueError(f"Transformer element '{tx_elem}' could not be activated")

        buses = dss_instance.CktElement.BusNames()
        hv_bus = buses[0].split(".")[0]

        freq = float(dss_instance.Solution.Frequency())

        # 1. Reset mode to snap and solve steady-state operating point to extract V_pre and I_pre
        dss_instance.Text.Command("solve mode=snap")
        dss_instance.Solution.Solve()
        v_pre_hv = self._extract_bus_complex_voltages(dss_instance, hv_bus)

        if not dss_instance.Circuit.SetActiveElement(tx_elem):
            raise ValueError(f"Transformer element '{tx_elem}' could not be activated")
        i_raw = dss_instance.CktElement.Currents()

        i_pre_hv = np.array(
            [
                complex(i_raw[0], i_raw[1]),
                complex(i_raw[2], i_raw[3]),
                complex(i_raw[4], i_raw[5]),
            ],
            dtype=complex,
        )

        # 2. Extract Z_sc matrix in faultstudy mode
        z_th_hv = self._extract_bus_zsc_matrix(dss_instance, hv_bus)
        v_th_hv = v_pre_hv + (z_th_hv @ i_pre_hv)

        thevenin = ThreePhaseThevenin(
            v_th=v_th_hv,
            z_th=z_th_hv,
            v_pre=v_pre_hv,
            i_pre=i_pre_hv,
            frequency_hz=freq,
        )
        thevenin.validate_equivalence()
        return thevenin

    def reduce_downstream_thevenin(
        self, dss_instance: Any, test_bus: str, tx_name: str = "trans1"
    ) -> ThreePhaseThevenin:
        """
        Calculates 3-phase Thévenin equivalent (V_th_LV, Z_th_LV) looking downstream into the base LV network
        at test_bus, explicitly excluding test load/fault elements.
        """
        # Ensure test fault elements are disabled in OpenDSS during reduction
        dss_instance.run_command("disable Fault.*")

        tx_elem = f"Transformer.{tx_name}"
        if not dss_instance.Circuit.SetActiveElement(tx_elem):
            raise ValueError(f"Transformer element '{tx_elem}' could not be activated")

        freq = float(dss_instance.Solution.Frequency())

        # 1. Reset OpenDSS to power flow snap mode to solve steady-state V_pre and I_pre
        dss_instance.Text.Command("solve mode=snap")
        dss_instance.Solution.Solve()
        v_pre_lv = self._extract_bus_complex_voltages(dss_instance, test_bus)

        # Query currents leaving transformer LV terminal (terminal 2) into downstream load network
        if not dss_instance.Circuit.SetActiveElement(tx_elem):
            raise ValueError(f"Transformer element '{tx_elem}' could not be activated")
        i_raw = dss_instance.CktElement.Currents()
        num_cond = dss_instance.CktElement.NumConductors()

        t2_start = 2 * num_cond

        i_pre_lv = np.array(
            [
                complex(i_raw[t2_start], i_raw[t2_start + 1]),
                complex(i_raw[t2_start + 2], i_raw[t2_start + 3]),
                complex(i_raw[t2_start + 4], i_raw[t2_start + 5]),
            ],
            dtype=complex,
        )

        # Calculate downstream base LV network load impedance matrix Z_th_LV = V_pre_LV / I_pre_LV
        z_th_lv = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            if abs(i_pre_lv[i]) <= 1e-6:
                raise ValueError(
                    f"Transformer LV terminal phase {i} current is zero ({i_pre_lv[i]} A), cannot compute downstream Thévenin load impedance"
                )
            z_val = v_pre_lv[i] / i_pre_lv[i]
            r_val = abs(float(np.real(z_val)))
            x_val = abs(float(np.imag(z_val)))
            z_th_lv[i, i] = complex(r_val, x_val)

        v_th_lv = v_pre_lv

        thevenin = ThreePhaseThevenin(
            v_th=v_th_lv,
            z_th=z_th_lv,
            v_pre=v_pre_lv,
            i_pre=i_pre_lv,
            frequency_hz=freq,
        )
        return thevenin

    def resolve_event_ports(
        self, dss_instance: Any, event: Any, feeder_idx: int = 1
    ) -> dict:
        """
        Resolves event targets into physical transformer HV/LV buses and LV test port.
        """
        tx_name = f"trans{feeder_idx}"
        tx_elem = f"Transformer.{tx_name}"
        if not dss_instance.Circuit.SetActiveElement(tx_elem):
            raise ValueError(
                f"Transformer '{tx_elem}' could not be activated in OpenDSS"
            )

        buses = dss_instance.CktElement.BusNames()
        hv_bus = buses[0].split(".")[0]
        lv_bus = buses[1].split(".")[0]

        # Test bus is the LV transformer secondary bus where LV network connects
        test_bus = lv_bus

        return {
            "tx_name": tx_name,
            "hv_bus": hv_bus,
            "lv_bus": lv_bus,
            "test_bus": test_bus,
            "feeder_idx": feeder_idx,
        }
