from opendssdirect import dss
from src.power_plant.plant import (
    generate_known_radial_topology,
    identify_candidate_consumer_meters,
    select_metered_consumers
)

# Aliases for backward compatibility
generate_radial_topology = generate_known_radial_topology
identify_candidate_pccs = identify_candidate_consumer_meters
select_metered_pccs = select_metered_consumers

def build_known_downstream_network(feeder_idx: int, topology: dict = None, loads_dict: dict = None):
    """
    Constructs the known LV distribution network in OpenDSS connected to the transformer secondary bus.
    """
    if topology is None:
        topology = generate_known_radial_topology(feeder_idx)

    # 1. Define linecodes for LV distribution lines (150 mm2 AAC overhead conductor, 350 A rating)
    dss.run_command(
        f"new linecode.lv_line_{feeder_idx} "
        f"nphases=3 "
        f"r1=0.21 x1=0.08 r0=0.63 x0=0.24 "
        f"c1=10.0 c0=5.0 units=km normamps=350.0"
    )

    # 2. Build LV lines
    for ln in topology.get("lines", []):
        r1 = ln.get("r1", 0.21)
        x1 = ln.get("x1", 0.08)
        r0 = ln.get("r0", 0.63)
        x0 = ln.get("x0", 0.24)
        length = ln.get("length", 0.05)

        dss.run_command(
            f"new line.{ln['name']} "
            f"bus1={ln['bus1']} bus2={ln['bus2']} "
            f"phases=3 "
            f"r1={r1} x1={x1} r0={r0} x0={x0} "
            f"length={length} units=km normamps=350.0"
        )

    # 3. Build consumer loads
    if loads_dict:
        for ld in loads_dict.get("loads", []):
            dss.run_command(
                f"new load.{ld['name']} "
                f"bus1={ld['bus']} "
                f"phases=3 "
                f"kv=0.415 "
                f"kw={ld['kw']} "
                f"pf={ld['pf']} "
                f"model={ld.get('model', 1)}"
            )

        for cap in loads_dict.get("capacitors", []):
            dss.run_command(
                f"new capacitor.{cap['name']} "
                f"bus1={cap['bus']} "
                f"phases=3 "
                f"kvar={cap['kvar']} "
                f"kv=0.415"
            )

        for mtr in loads_dict.get("motors", []):
            dss.run_command(
                f"new load.{mtr['name']} "
                f"bus1={mtr['bus']} "
                f"phases=3 "
                f"kv=0.415 "
                f"kw={mtr['kw']} "
                f"pf={mtr['pf']} "
                f"model=1"
            )

# Alias for backward compatibility
build_hidden_downstream_network = build_known_downstream_network

def update_downstream_loads(topology: dict, load_scale: float = 1.0, cap_state: bool = True):
    """
    Updates the active power demands of consumer loads across the known network.
    """
    for ld in topology.get("loads", []):
        p_scaled = ld["kw"] * load_scale
        dss.run_command(f"edit load.{ld['name']} kw={round(p_scaled, 2)}")

    for cap in topology.get("capacitors", []):
        dss.run_command(f"edit capacitor.{cap['name']} enabled={'yes' if cap_state else 'no'}")
