"""
Explicit Physical 11/0.415 kV LV Distribution Transformer Models.
Contains 3 distinct transformer models for feeder edge interfaces.
"""

import numpy as np
from src.transient.models import (
    TransformerSpec,
    TransformerWinding,
    ShortCircuitTest,
)

# Explicit LV Distribution Transformer Specifications Matrix
TRANSFORMER_MODELS = {
    "trans1": {
        "spec_id": "tx_spec_std_1500kva",
        "name": "trans1",
        "phases": 3,
        "windings": 2,
        "conns": ["delta", "wye"],
        "kvs": [11.0, 0.415],
        "kvas": [1500.0, 1500.0],
        # Winding & Losses
        "r_pct": 0.60,         # 0.60% resistance (50 kW copper loss @ full load)
        "xhl_pct": 8.33,       # 8.33% leakage reactance (8.35% %Z, X/R = 13.9)
        "noloadloss_pct": 0.1, # 7.5 kW core loss (0.1%)
        "loadloss_pct": 0.667, # 50 kW copper loss (0.667%)
        "imag_pct": 0.8,       # 0.8% excitation current
        # Sequence Impedances (pu)
        "r0_pct": 1.20,        # Zero-sequence resistance 0.0120 pu
        "x0_pct": 4.50,        # Zero-sequence reactance 0.0450 pu
        "rm_pu": 800.0,        # Core-loss resistance 800 pu
        "xm_pu": 250.0         # Magnetizing reactance 250 pu
    },
    "trans2": {
        "spec_id": "tx_spec_high_z_1200kva",
        "name": "trans2",
        "phases": 3,
        "windings": 2,
        "conns": ["delta", "wye"],
        "kvs": [11.0, 0.415],
        "kvas": [1200.0, 1200.0],
        "r_pct": 0.80,
        "xhl_pct": 10.0,
        "noloadloss_pct": 0.12,
        "loadloss_pct": 0.75,
        "imag_pct": 1.0,
        "r0_pct": 1.60,
        "x0_pct": 5.50,
        "rm_pu": 750.0,
        "xm_pu": 220.0
    },
    "trans3": {
        "spec_id": "tx_spec_low_loss_2000kva",
        "name": "trans3",
        "phases": 3,
        "windings": 2,
        "conns": ["delta", "wye"],
        "kvs": [11.0, 0.415],
        "kvas": [2000.0, 2000.0],
        "r_pct": 0.40,
        "xhl_pct": 6.5,
        "noloadloss_pct": 0.08,
        "loadloss_pct": 0.50,
        "imag_pct": 0.6,
        "r0_pct": 0.80,
        "x0_pct": 3.50,
        "rm_pu": 900.0,
        "xm_pu": 300.0
    }
}

# Single Baseline Transformer Model for Datasets 2 and 3
BASELINE_TRANSFORMER_MODEL = TRANSFORMER_MODELS["trans1"]


def get_distribution_transformer_spec(feeder_idx: int, use_baseline: bool = False) -> dict:
    """
    Returns the physical specifications of the 11/0.415 kV distribution step-down transformer.

    Args:
        feeder_idx: Feeder index (1, 2, or 3).
        use_baseline: If True, uses the single baseline transformer model across all feeders.
                     If False, returns the feeder-specific distinct transformer model.
    """
    tx_key = f"trans{feeder_idx}"
    if use_baseline:
        base_spec = BASELINE_TRANSFORMER_MODEL.copy()
        base_spec["name"] = tx_key
        base_spec["buses"] = [f"feeder{feeder_idx}_head", f"feeder{feeder_idx}_sec"]
        return base_spec

    spec = TRANSFORMER_MODELS.get(tx_key, TRANSFORMER_MODELS["trans1"]).copy()
    spec["name"] = tx_key
    spec["buses"] = [f"feeder{feeder_idx}_head", f"feeder{feeder_idx}_sec"]
    return spec


def build_transformer_spec(
    feeder_idx: int, use_baseline: bool = False, frequency_hz: float = 50.0
) -> TransformerSpec:
    """
    Constructs and returns the domain TransformerSpec object for a given feeder.
    """
    spec_dict = get_distribution_transformer_spec(feeder_idx, use_baseline=use_baseline)
    kvas = spec_dict["kvas"]
    kvs = spec_dict["kvs"]
    r_pct = float(spec_dict["r_pct"])
    xhl_pct = float(spec_dict["xhl_pct"])
    noloadloss_pct = float(spec_dict["noloadloss_pct"])
    imag_pct = float(spec_dict["imag_pct"])
    conns = spec_dict.get("conns", ["delta", "wye"])

    r0_pct = float(spec_dict.get("r0_pct", r_pct))
    x0_pct = float(spec_dict.get("x0_pct", xhl_pct))
    z0_pu = float(np.sqrt((r0_pct / 100.0) ** 2 + (x0_pct / 100.0) ** 2))
    losses_zero_kw = (r0_pct / 100.0) * float(kvas[0])

    return TransformerSpec(
        name=str(spec_dict["name"]),
        frequency_hz=float(frequency_hz),
        windings=[
            TransformerWinding("HV", float(kvs[0]), float(kvas[0]) / 1000.0, str(conns[0]), 0.0),
            TransformerWinding("LV", float(kvs[1]), float(kvas[1]) / 1000.0, str(conns[1]), -30.0),
        ],
        short_circuit_tests=[
            ShortCircuitTest(
                1,
                2,
                z_pos_pu=float(np.sqrt((r_pct / 100.0) ** 2 + (xhl_pct / 100.0) ** 2)),
                losses_pos_kw=(r_pct / 100.0) * float(kvas[0]),
                z_zero_pu=z0_pu,
                losses_zero_kw=losses_zero_kw,
            )
        ],
        excitation_current_percent=imag_pct,
        excitation_loss_kw=(noloadloss_pct / 100.0) * float(kvas[0]),
    )
