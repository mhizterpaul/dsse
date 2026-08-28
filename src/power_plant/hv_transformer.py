"""
Upstream Distribution Substation HV Transformer Parameters & OpenDSS Builder.
Aligns with specs in docs/specs/upstream_transformer.md.
"""

# Upstream Distribution Substation Transformer Specification Matrix (33/11-kV)
HV_TRANSFORMER_SPEC = {
    "transformer_id": "substation",
    "name": "hv_substation_tx",
    "phases": 3,
    "windings": 2,
    "conns": ["delta", "wye"],
    "kvs": [33.0, 11.0],
    "kvas": [7500.0, 7500.0],
    # Winding & Losses
    "r_pct": 0.60,          # 0.60% resistance (50 kW copper loss @ full load)
    "xhl_pct": 8.33,        # 8.33% leakage reactance (8.35% %Z)
    "noloadloss_pct": 0.1,  # 0.10% core loss (7.5 kW core loss)
    "loadloss_pct": 0.667,  # 0.667% copper loss (50 kW)
    "imag_pct": 0.4,        # 0.4% excitation current
    # Core & Magnetizing parameters
    "rm_pu": 800.0,         # Core-loss resistance 800 pu
    "xm_pu": 250.0,         # Magnetizing reactance 250 pu
    "buses": ["sourcebus", "main_bus"]
}


def build_hv_transformer(dss, spec: dict = None, verbose: bool = False) -> dict:
    """
    Builds and configures the 33/11 kV upstream distribution substation transformer in OpenDSS.

    Args:
        dss: The single OpenDSS direct instance passed from runner.
        spec: Optional custom specification dict; defaults to HV_TRANSFORMER_SPEC.
        verbose: Controls whether progress log statement is printed (default: False).
    """
    if spec is None:
        spec = HV_TRANSFORMER_SPEC

    cmd = (
        f"new transformer.{spec['name']} "
        f"phases={spec['phases']} windings={spec['windings']} "
        f"buses=[{','.join(spec['buses'])}] "
        f"conns=[{','.join(spec['conns'])}] "
        f"kvs=[{','.join(map(str, spec['kvs']))}] "
        f"kvas=[{','.join(map(str, spec['kvas']))}] "
        f"%r={spec['r_pct']} "
        f"xhl={spec['xhl_pct']} "
        f"%noloadloss={spec['noloadloss_pct']} "
        f"%imag={spec['imag_pct']}"
    )
    dss.run_command(cmd)
    if verbose:
        print(f"INFO: Built HV Substation Transformer '{spec['name']}' (33/11 kV, 7.5 MVA)")
    return spec
