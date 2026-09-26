import os
import sys
import math
from pathlib import Path
from typing import Optional, Union, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import schemdraw
import schemdraw.elements as elm

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


def plot_opendss_circuit_matplotlib_fallback(
    dss,
    bus_coords: Dict[str, tuple[float, float]],
    dots: bool = True,
    labels: bool = False,
) -> plt.Figure:
    """
    Render the active OpenDSS circuit directly with Matplotlib as fallback.
    Evaluated strictly when native OpenDSS circuit plot is empty or unavailable.
    Geometry and connectivity come directly from bus_coords and active OpenDSS elements.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    missing = []

    def bus_name(raw_bus: str) -> str:
        return raw_bus.split(".")[0].strip().lower()

    def get_element_buses(element_class, element_name):
        element_class.Name(element_name)
        return [bus_name(bus) for bus in dss.CktElement.BusNames()]

    # 1. Distribution lines
    for line_name in dss.Lines.AllNames():
        buses = get_element_buses(dss.Lines, line_name)
        if len(buses) < 2:
            continue
        b1, b2 = buses[0], buses[1]
        if b1 not in bus_coords or b2 not in bus_coords:
            missing.append(f"Line.{line_name}: {b1} -> {b2}")
            continue

        x1, y1 = bus_coords[b1]
        x2, y2 = bus_coords[b2]
        ax.plot([x1, x2], [y1, y2], linewidth=1.8, color="steelblue", solid_capstyle="round", zorder=1)

    # 2. Transformers
    for tx_name in dss.Transformers.AllNames():
        buses = get_element_buses(dss.Transformers, tx_name)
        if len(buses) < 2:
            continue
        b0 = buses[0]
        if b0 not in bus_coords:
            missing.append(f"Transformer.{tx_name}: {b0}")
            continue

        x0, y0 = bus_coords[b0]
        for b in buses[1:]:
            if b not in bus_coords:
                missing.append(f"Transformer.{tx_name}: {b}")
                continue
            x, y = bus_coords[b]
            ax.plot([x0, x], [y0, y], linewidth=2.8, color="crimson", zorder=2)

    # 3. Bus dots
    if dots:
        xs, ys = [], []
        for bus, (x, y) in bus_coords.items():
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, s=20, color="navy", zorder=5)

    # 4. Generator buses
    generator_buses = set()
    for gen_name in dss.Generators.AllNames():
        dss.Generators.Name(gen_name)
        buses = dss.CktElement.BusNames()
        if buses:
            generator_buses.add(bus_name(buses[0]))

    for bus in generator_buses:
        if bus in bus_coords:
            x, y = bus_coords[bus]
            ax.scatter([x], [y], marker="^", color="darkgreen", s=140, zorder=10, label="Generator")

    # 5. Load buses
    load_buses = set()
    for load_name in dss.Loads.AllNames():
        dss.Loads.Name(load_name)
        buses = dss.CktElement.BusNames()
        if buses:
            load_buses.add(bus_name(buses[0]))

    for bus in load_buses:
        if bus in bus_coords:
            x, y = bus_coords[bus]
            ax.scatter([x], [y], marker="o", color="royalblue", s=45, zorder=8, label="Load")

    ax.set_title("OpenDSS Distribution Network Circuit Plot", fontsize=13, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    return fig


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
    Initializes OpenDSS plant session and generates OpenDSS circuit plot using plant parameters
    directly from plant.py with schemdraw icons and proper labels for lines, loads, generator, and transformers.
    Strictly evaluates fallback method only when native DSS plot is empty or unavailable.
    Starts load labeling from the end of the line, applying offsets for labels close to the beginning.
    """
    from dss import plot as dss_plot

    runner = CoSimulationRunner()
    if dss is None:
        dss = runner.dss

    plant_data = runner.initialize_plant_session(use_baseline_feeder=use_baseline_transformers, seed=42)

    # 1. Assign spatial coordinates dynamically for active feeder topologies
    bus_coords = assign_bus_coordinates_from_plant(dss, plant_data)

    # 2. Enable DSS-Extensions plotting with show=False to preserve Matplotlib figure
    dss_plot.enable(
        plot2d=True,
        plot3d=False,
        show=False,
    )

    dss.run_command("Set Voltagebases=[33.0, 11.0, 0.415]")
    dss.run_command("CalcVoltageBases")
    dss.run_command("solve")

    if mark_transformers:
        dss.run_command("Set MarkTransformers=Y")
    if mark_regulators:
        dss.run_command("Set MarkRegulators=Y")

    dots_str = "Y" if dots else "N"
    cmd = f"Plot Circuit Quantity={quantity} Dots={dots_str} Labels=N Subs=N"

    fig = None
    ax = None
    native_plot_error = None

    figs_before = set(plt.get_fignums())

    try:
        dss.run_command(cmd)

        figs_after = set(plt.get_fignums())
        new_figs = sorted(figs_after - figs_before)

        if new_figs:
            candidate = plt.figure(new_figs[-1])
            if candidate.axes:
                candidate_ax = candidate.axes[0]
                total_elements = (
                    len(candidate_ax.collections)
                    + len(candidate_ax.lines)
                    + len(candidate_ax.patches)
                )
                if total_elements > 0:
                    fig = candidate
                    ax = candidate_ax
                    while ax.texts:
                        ax.texts[0].remove()
    except Exception as exc:
        native_plot_error = exc

    # Strictly evaluate fallback ONLY when image/figure generated by DSS plotter is empty or failed
    if fig is None or ax is None:
        print("Native OpenDSS plotting generated an empty image or was unavailable; evaluating Matplotlib fallback.")
        if native_plot_error is not None:
            print(f"Native plotting error: {native_plot_error}")

        fig = plot_opendss_circuit_matplotlib_fallback(
            dss=dss,
            bus_coords=bus_coords,
            dots=dots,
            labels=labels,
        )
        ax = fig.axes[0]

    # 3. Add standard schemdraw element icons & legible text labels
    if "generator_info" not in plant_data or "generator_kw" not in plant_data["generator_info"]:
        raise KeyError("Missing required 'generator_info.generator_kw' in plant_data.")
    gen_kw = plant_data["generator_info"]["generator_kw"]

    topologies = plant_data["topology"]["topologies"]
    active_feeder_ids = list(topologies.keys())

    with schemdraw.Drawing(canvas=ax) as d:
        # Substation Generator / Source
        if "sourcebus" in bus_coords:
            sx, sy = bus_coords["sourcebus"]
            d.add(elm.SourceSin().at((sx - 35, sy)).scale(0.55))
            ax.text(
                sx + 15, sy - 15,
                f"Source (33 kV Generator, {gen_kw/1000.0:.1f} MW)",
                fontsize=9, fontweight="bold", color="darkgreen", zorder=25
            )

        # Substation Upstream Transformer (33/11 kV)
        if "sourcebus" in bus_coords:
            sx, sy = bus_coords["sourcebus"]
            d.add(elm.Transformer().at((sx, sy + 35)).scale(0.55))
            ax.text(
                sx + 20, sy + 50,
                "Trxr 33/11 kV",
                fontsize=9, fontweight="bold", color="crimson", zorder=25
            )

        # Downstream Distribution Transformer(s) (11/0.415 kV) - exactly 1 for Dataset 1
        for f_id in active_feeder_ids:
            fbus = f"feeder{f_id}_head"
            if fbus in bus_coords:
                tx, ty = bus_coords[fbus]
                d.add(elm.Transformer().at((tx, ty)).scale(0.55))
                ax.text(
                    tx + 22, ty + 10,
                    f"Trxr 11/0.415 kV Tx{f_id}",
                    fontsize=9, fontweight="bold", color="darkred", zorder=25
                )

        # Consumer Loads: Labeling starts from the END of the transmission/distribution line
        if "registry" not in plant_data:
            raise KeyError("Missing required 'registry' in plant_data.")
        registry = plant_data["registry"]

        source_x, source_y = bus_coords.get("sourcebus", (0.0, 0.0))
        all_consumers = registry.get_all_consumers()

        def get_dist_from_source(unit):
            b_name = unit.bus_id.lower()
            if b_name in bus_coords:
                bx, by = bus_coords[b_name]
                return math.hypot(bx - source_x, by - source_y)
            return 0.0

        # Sort consumers in descending order of distance (furthest/end of line first)
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

                    d.add(elm.RBox().at((bx, by)).scale(0.35))

                    formatted_label = ltype.replace("_", " ").title()

                    # Apply position offset to labels close to the beginning of the transmission line (dist < 180)
                    if dist < 180.0:
                        offset_x = 35.0
                        offset_y = 20.0
                    else:
                        offset_x = 8.0
                        offset_y = 8.0

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

    # Dummy plot handles for legend
    ax.plot([], [], "-", color="steelblue", linewidth=2, label="Distribution Lines")

    handles, legend_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=9
    )

    ax.set_title("OpenDSS Distribution Network Circuit Plot", fontsize=12, fontweight="bold")

    # Save plot
    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, bbox_inches="tight", dpi=300)

    # 4. Validate plot non-emptiness
    validate_network_plot(fig, save_path=save_path)

    return fig


if __name__ == "__main__":
    plot_opendss_circuit(save_path="src/visualization/dss_circuit_plot.png")
    print("OpenDSS circuit plot generated, validated, and saved successfully.")
