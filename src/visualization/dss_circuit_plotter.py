import os
import sys
import math
from pathlib import Path
from typing import Optional, Union, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pypsa

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner


def get_bus_voltage_angle_deg(dss_instance, bus_name: str) -> float:
    """
    Queries OpenDSS directly for the active Phase-A voltage phase angle (in degrees) of a specified bus.
    Raises ValueError if the bus cannot be activated or has no voltage solution data.
    """
    if not dss_instance.Circuit.SetActiveBus(bus_name):
        raise ValueError(f"Could not set active bus '{bus_name}' in OpenDSS circuit.")
    v_mag_ang = dss_instance.Bus.VMagAngle()
    if not v_mag_ang or len(v_mag_ang) < 2:
        raise ValueError(f"Bus '{bus_name}' in OpenDSS circuit has no solved voltage angle data.")
    return float(v_mag_ang[1])


def assign_bus_coordinates_from_plant(dss_instance, plant_data: Dict[str, Any]) -> Dict[str, tuple[float, float]]:
    """
    Dynamically computes and assigns spatial coordinates (X, Y) to active network buses in OpenDSS circuit.
    Retrieves voltage phase angles directly from solved OpenDSS model solution (dss_instance.Bus.VMagAngle())
    and line lengths from active OpenDSS line elements and plant_data topology parameters.
    Only computes coordinates for active feeders defined in plant_data['topology']['topologies'].
    """
    # Solve OpenDSS circuit first to populate voltage solution and phase angles
    dss_instance.run_command("Set Voltagebases=[33.0, 11.0, 0.415]")
    dss_instance.run_command("CalcVoltageBases")
    dss_instance.run_command("solve")

    buses = [b.lower() for b in dss_instance.Circuit.AllBusNames()]
    coords = {}

    # 1. Base Substation & Main Bus Coordinates
    coords["sourcebus"] = (0.0, 0.0)
    coords["main_bus"] = (0.0, 100.0)

    if "topology" not in plant_data or "topologies" not in plant_data["topology"]:
        raise KeyError("Missing required 'topology.topologies' structure in plant_data.")
    topologies = plant_data["topology"]["topologies"]
    active_feeder_ids = list(topologies.keys())

    # Dynamically extract MV feeder lengths and voltage phase angles directly from OpenDSS
    mv_feeder_lengths = {}
    feeder_angles = {}
    for f_id in active_feeder_ids:
        line_name = f"Line.mv_feeder_{f_id}"
        if not dss_instance.Circuit.SetActiveElement(line_name):
            raise ValueError(f"Required OpenDSS line element '{line_name}' not found in active circuit.")
        val = dss_instance.Properties.Value("length")
        if not val:
            raise ValueError(f"Line element '{line_name}' missing length property value.")
        mv_feeder_lengths[f_id] = float(val)

        head_bus = f"feeder{f_id}_head"
        feeder_angles[f_id] = get_bus_voltage_angle_deg(dss_instance, head_bus)

    # Calculate spatial position of feeder heads using line length and OpenDSS voltage phase angle
    for f_id, length in mv_feeder_lengths.items():
        ang_deg = feeder_angles[f_id]
        ang_rad = math.radians(ang_deg)
        dist = length * 30.0
        hx = coords["main_bus"][0] + dist * math.cos(ang_rad)
        hy = coords["main_bus"][1] + dist * math.sin(ang_rad)
        head_bus = f"feeder{f_id}_head"
        sec_bus = f"feeder{f_id}_sec"
        if head_bus in buses:
            coords[head_bus] = (hx, hy)
        if sec_bus in buses:
            coords[sec_bus] = (hx, hy + 30.0)

    # 2. Traverse LV Network Radial Topologies dynamically from plant_data
    for f_id, sub_topo in topologies.items():
        sec_bus = f"feeder{f_id}_sec"
        if sec_bus not in coords:
            raise KeyError(f"Secondary bus '{sec_bus}' not found in computed coordinates.")

        adj = {}
        if "lines" not in sub_topo:
            raise KeyError(f"Missing required 'lines' list in sub-topology for feeder {f_id}.")

        for ln in sub_topo["lines"]:
            if "bus1" not in ln or "bus2" not in ln or "length" not in ln:
                raise KeyError(f"Missing 'bus1', 'bus2', or 'length' parameter in line record: {ln}.")
            p = str(ln["bus1"]).lower()
            c = str(ln["bus2"]).lower()
            l = float(ln["length"])
            adj.setdefault(p, []).append((c, l))

        queue = [sec_bus]
        while queue:
            parent = queue.pop(0)
            px, py = coords[parent]
            children = adj.get(parent, [])
            for child, line_len in children:
                c_ang = get_bus_voltage_angle_deg(dss_instance, child)
                c_rad = math.radians(c_ang)
                scale = max(line_len * 1000.0, 30.0)
                cx = px + scale * math.cos(c_rad)
                cy = py + scale * math.sin(c_rad)
                coords[child] = (cx, cy)
                queue.append(child)

    # 3. Apply SetBusXY to OpenDSS for all active buses present in coords
    for b in dss_instance.Circuit.AllBusNames():
        b_lower = b.lower()
        if b_lower in coords:
            x, y = coords[b_lower]
            dss_instance.run_command(f"SetBusXY Bus={b} x={x} y={y}")

    return coords


def build_pypsa_network_from_opendss(dss, plant_data: Dict[str, Any]) -> pypsa.Network:
    """
    Constructs a PyPSA Network instance populated dynamically from active OpenDSS circuit elements
    and plant_data topology parameters without hardcoded fallback assumptions.
    Raises KeyError or ValueError if required network elements or properties are missing.
    """
    bus_coords = assign_bus_coordinates_from_plant(dss, plant_data)

    net = pypsa.Network()

    def clean_bus_name(raw_bus: str) -> str:
        return raw_bus.split(".")[0].strip().lower()

    # 1. Add Buses with nominal voltages queried directly from OpenDSS bus voltage bases
    for bus_name in dss.Circuit.AllBusNames():
        b_clean = bus_name.lower()
        if b_clean in bus_coords:
            bx, by = bus_coords[b_clean]

            dss.Circuit.SetActiveBus(bus_name)
            kv_base = float(dss.Bus.kVBase())
            if kv_base <= 0:
                err_msg = f"Invalid base voltage evaluated for bus '{bus_name}': {kv_base} kV"
                print(f"ERROR: {err_msg}")
                raise ValueError(err_msg)

            v_nom = kv_base * (3.0 ** 0.5)
            net.add("Bus", b_clean, x=bx, y=by, v_nom=v_nom)

    # 2. Add Generator with capacity retrieved directly from plant_data
    if "generator_info" not in plant_data or "generator_kw" not in plant_data["generator_info"]:
        err_msg = "Missing required 'generator_info.generator_kw' in plant_data"
        print(f"ERROR: {err_msg}")
        raise KeyError(err_msg)

    gen_kw = float(plant_data["generator_info"]["generator_kw"])
    if gen_kw <= 0:
        err_msg = f"Invalid generator capacity in plant_data: {gen_kw} kW"
        print(f"ERROR: {err_msg}")
        raise ValueError(err_msg)

    if "sourcebus" not in net.buses.index:
        err_msg = "Substation 'sourcebus' missing from PyPSA network buses"
        print(f"ERROR: {err_msg}")
        raise KeyError(err_msg)

    net.add("Generator", "gen_33kv", bus="sourcebus", p_nom=gen_kw/1000.0)

    # 3. Add Lines with physical properties queried directly from OpenDSS active elements
    for line_name in dss.Lines.AllNames():
        if not dss.Circuit.SetActiveElement(f"Line.{line_name}"):
            continue

        buses = [clean_bus_name(b) for b in dss.CktElement.BusNames()]
        if len(buses) < 2 or buses[0] not in net.buses.index or buses[1] not in net.buses.index:
            continue

        b1, b2 = buses[0], buses[1]

        length = float(dss.Properties.Value("length"))
        r1 = float(dss.Properties.Value("r1"))
        x1 = float(dss.Properties.Value("x1"))

        r_ohm = r1 * length
        x_ohm = x1 * length

        if length <= 0 or r_ohm <= 0:
            err_msg = f"Invalid line parameters for '{line_name}': length={length}, r={r_ohm}"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

        net.add("Line", line_name, bus0=b1, bus1=b2, length=length, r=r_ohm, x=x_ohm)

    # 4. Add Transformers with MVA ratings queried directly from OpenDSS properties
    for tx_name in dss.Transformers.AllNames():
        if not dss.Circuit.SetActiveElement(f"Transformer.{tx_name}"):
            continue

        buses = [clean_bus_name(b) for b in dss.CktElement.BusNames()]
        if len(buses) < 2 or buses[0] not in net.buses.index or buses[1] not in net.buses.index:
            continue

        mva_val = float(dss.Properties.Value("kVA")) / 1000.0
        if mva_val <= 0:
            err_msg = f"Invalid transformer capacity rating for '{tx_name}': {mva_val} MVA"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

        net.add("Transformer", tx_name, bus0=buses[0], bus1=buses[1], s_nom=mva_val)

    # 5. Add Consumer Loads with kW power demands queried directly from OpenDSS active loads
    for load_name in dss.Loads.AllNames():
        if not dss.Circuit.SetActiveElement(f"Load.{load_name}"):
            continue

        b_load = clean_bus_name(dss.CktElement.BusNames()[0])
        if b_load not in net.buses.index:
            continue

        kw = float(dss.Properties.Value("kW"))
        if kw < 0:
            err_msg = f"Invalid negative power demand for load '{load_name}': {kw} kW"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

        net.add("Load", load_name, bus=b_load, p_set=kw/1000.0)

    return net


def validate_network_plot(fig: plt.Figure, save_path: Optional[Union[str, Path]] = None):
    """
    Validates that the generated OpenDSS distribution network plot is not empty.
    Raises ValueError if empty.
    """
    if not fig or not fig.axes:
        raise ValueError("Generated OpenDSS distribution network plot figure or axes is empty.")

    ax = fig.axes[0]
    total_elements = len(ax.collections) + len(ax.lines) + len(ax.patches) + len(ax.texts)
    if total_elements == 0:
        raise ValueError("OpenDSS distribution network plot contains no graphical elements or network lines.")

    if save_path and Path(save_path).exists():
        img = Image.open(save_path)
        arr = np.array(img)
        if arr.size == 0:
            raise ValueError(f"Saved distribution network plot at '{save_path}' is empty (0 bytes or size).")

        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            non_white = np.sum(np.mean(arr[:, :, :3], axis=2) < 250)
            if non_white < 500:
                raise ValueError(
                    f"Saved distribution network plot image '{save_path}' appears visually empty "
                    f"(non-white pixel count: {non_white})."
                )


def plot_opendss_circuit(
    dss=None,
    use_baseline_transformers: bool = True,
    quantity: str = "Power",
    dots: bool = True,
    labels: bool = False,
    subs: bool = True,
    mark_transformers: bool = True,
    mark_regulators: bool = True,
    save_path: str = "src/visualization/dss_circuit_plot.png"
):
    """
    Initializes OpenDSS plant session, builds a PyPSA Network representation, and renders the single-line diagram
    using PyPSA's network plotting engine (`net.plot`).
    Starts load labeling from the end of the line, applying offsets for labels close to the beginning.
    """
    runner = CoSimulationRunner()
    if dss is None:
        dss = runner.dss

    plant_data = runner.initialize_plant_session(use_baseline_feeder=use_baseline_transformers, seed=42)

    # 1. Build PyPSA Network instance directly from active OpenDSS session and plant_data
    net = build_pypsa_network_from_opendss(dss, plant_data)

    fig, ax = plt.subplots(figsize=(14, 10))

    # 2. Render PyPSA single line diagram with geomap=False to target standard Matplotlib axes
    net.plot(
        ax=ax,
        geomap=False,
        bus_sizes=0.015 if dots else 0.0,
        bus_colors="navy",
        line_colors="steelblue",
        line_widths=2.0,
        transformer_colors="crimson",
        title="Distribution Network Single Line Diagram (PyPSA)",
    )

    # 3. Legible consumer load and equipment text labels
    if "registry" not in plant_data:
        raise KeyError("Missing required 'registry' in plant_data.")
    registry = plant_data["registry"]

    bus_coords = {b: (net.buses.loc[b, "x"], net.buses.loc[b, "y"]) for b in net.buses.index}
    source_x, source_y = bus_coords.get("sourcebus", (0.0, 0.0))
    all_consumers = registry.get_all_consumers()

    def get_dist_from_source(unit):
        b_name = unit.bus_id.lower()
        if b_name in bus_coords:
            bx, by = bus_coords[b_name]
            return math.hypot(bx - source_x, by - source_y)
        return 0.0

    sorted_consumers = sorted(all_consumers, key=get_dist_from_source, reverse=True)

    labeled_types = set()
    for unit in sorted_consumers:
        b_name = unit.bus_id.lower()
        if b_name not in bus_coords:
            continue
        bx, by = bus_coords[b_name]
        dist = math.hypot(bx - source_x, by - source_y)

        for ld in unit.loads:
            ltype = ld.load_type
            if ltype not in labeled_types:
                labeled_types.add(ltype)
                formatted_label = ltype.replace("_", " ").title()

                offset_x = 35.0 if dist < 180.0 else 8.0
                offset_y = 20.0 if dist < 180.0 else 8.0

                ax.text(
                    bx + offset_x, by + offset_y,
                    formatted_label,
                    fontsize=8, fontweight="bold", color="navy",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="lightyellow",
                        edgecolor="royalblue",
                        alpha=0.9
                    ),
                    zorder=26
                )

    ax.set_title("OpenDSS Distribution Network Single Line Diagram (PyPSA)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    # Save plot
    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, bbox_inches="tight", dpi=300)

    # Validate plot non-emptiness
    validate_network_plot(fig, save_path=save_path)

    return fig


if __name__ == "__main__":
    plot_opendss_circuit(save_path="src/visualization/dss_circuit_plot.png")
    print("PyPSA distribution network single line diagram generated, validated, and saved successfully.")
