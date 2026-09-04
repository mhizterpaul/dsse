import pytest
import numpy as np
from opendssdirect import dss
import src.power_plant.plant as plant
from src.transient.opendss_reducer import OpenDSSReducer


def test_opendss_reducer_port_resolution():
    plant_data = plant.build_single_lv_network_composition(dss, seed=42)
    reducer = OpenDSSReducer()

    ports = reducer.resolve_event_ports(dss, None, feeder_idx=1)
    assert ports["tx_name"] == "trans1"
    assert ports["hv_bus"] == "feeder1_head"
    assert ports["test_bus"] == "feeder1_sec"


def test_opendss_reducer_reductions():
    plant_data = plant.build_single_lv_network_composition(dss, seed=42)
    reducer = OpenDSSReducer()

    up_th = reducer.reduce_upstream_thevenin(dss, tx_name="trans1")
    assert up_th.v_th.shape == (3,)
    assert up_th.z_th.shape == (3, 3)
    assert np.all(np.abs(up_th.v_th) > 5000.0)

    down_th = reducer.reduce_downstream_thevenin(dss, test_bus="feeder1_sec", tx_name="trans1")
    assert down_th.v_th.shape == (3,)
    assert down_th.z_th.shape == (3, 3)
    assert np.all(np.abs(down_th.v_th) > 200.0)

    # Verify reconstruction equivalence
    assert down_th.validate_equivalence()
