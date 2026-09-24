import os
import sys
import math
from pathlib import Path
from typing import Optional, Union, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
    Dynamically computes and assigns spatial coordinates (X, Y) to all buses in the active OpenDSS circuit.
    Retrieves voltage phase angles directly from the solved OpenDSS model solution (dss_instance.Bus.VMagAngle())
    and line lengths from active OpenDSS line elements and plant_data topology parameters.
    Raises KeyError or ValueError if required topology or element parameters are missing (no default fallbacks permitted).
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

    # Dynamically extract MV feeder lengths and voltage phase angles directly from OpenDSS
    mv_feeder_lengths = {}
    feeder_angles = {}
    for f_id in [1, 2, 3]:
        line_name = f"Line.mv_feeder_{f_id}"
        if not dss_instance.Circuit.SetActiveElement(line_name):
            raise ValueError(f"Required OpenDSS line element '{line_name}' not found in active circuit.")
        val = dss_instance.Properties.Value("length")
        if not val:
            raise ValueError(f"Line element '{line_name}' missing length property value.")
        mv_feeder_lengths[f_id] = float(val)

        head_bus = f"feeder{f_id}_head"
        # Query voltage phase angle directly from OpenDSS model
        feeder_angles[f_id] = get_bus_voltage_angle_deg(dss_instance, head_bus)

    # Calculate spatial position of feeder heads using line length and OpenDSS voltage phase angle
    for f_id, length in mv_feeder_lengths.items():
        # Retrieve angle directly from OpenDSS solved phase angle
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
    if "topology" not in plant_data or "topologies" not in plant_data["topology"]:
        raise KeyError("Missing required 'topology.topologies' structure in plant_data.")
    topologies = plant_data["topology"]["topologies"]

    for f_id, sub_topo in topologies.items():
        sec_bus = f"feeder{f_id}_sec"
        if sec_bus not in coords:
            raise KeyError(f"Secondary bus '{sec_bus}' not found in computed coordinates.")

        # Build adjacency graph from topology line parameters
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

        # BFS/DFS traversal using OpenDSS voltage phase angles directly queried per bus
        queue = [sec_bus]

        while queue:
            parent = queue.pop(0)
            px, py = coords[parent]
            p_ang = get_bus_voltage_angle_deg(dss_instance, parent)
            children = adj.get(parent, [])
            for child, line_len in children:
                # Query child bus phase angle directly from OpenDSS model
                c_ang = get_bus_voltage_angle_deg(dss_instance, child)
                c_rad = math.radians(c_ang)
                scale = max(line_len * 1000.0, 30.0)
                cx = px + scale * math.cos(c_rad)
                cy = py + scale * math.sin(c_rad)
                coords[child] = (cx, cy)
                queue.append(child)

    # 3. Apply SetBusXY to OpenDSS for all circuit buses; raise KeyError if any bus lacks spatial coordinates
    for b in dss_instance.Circuit.AllBusNames():
        b_lower = b.lower()
        if b_lower not in coords:
            raise KeyError(f"Bus '{b}' in OpenDSS circuit has no computed spatial coordinates in topology.")
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
    Render the active OpenDSS circuit directly with Matplotlib.

    This is used only when the DSS-Python/OpenDSS Plot Circuit backend
    does not leave a usable Matplotlib figure behind.

    Geometry comes from bus_coords, while connectivity comes directly
    from the active OpenDSS Line and Transformer elements.
    """

    fig, ax = plt.subplots(figsize=(14, 10))

    missing = []

    def bus_name(raw_bus: str) -> str:
        # Convert e.g. feeder1_head.1.2.3 -> feeder1_head
        return raw_bus.split(".")[0].strip().lower()

    def get_element_buses(element_class, element_name):
        element_class.Name(element_name)
        return [
            bus_name(bus)
            for bus in dss.CktElement.BusNames()
        ]

    # ------------------------------------------------------------------
    # 1. Distribution lines
    # ------------------------------------------------------------------
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

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=1.8,
            solid_capstyle="round",
            zorder=1,
        )

    # ------------------------------------------------------------------
    # 2. Transformers
    # ------------------------------------------------------------------
    for tx_name in dss.Transformers.AllNames():
        buses = get_element_buses(dss.Transformers, tx_name)

        if len(buses) < 2:
            continue

        b0 = buses[0]

        if b0 not in bus_coords:
            missing.append(f"Transformer.{tx_name}: {b0}")
            continue

        x0, y0 = bus_coords[b0]

        # Two-winding transformer:
        # directly connect winding 1 -> winding 2.
        #
        # For multi-winding transformers, draw each winding from the
        # first winding bus as a simple star representation.
        for b in buses[1:]:
            if b not in bus_coords:
                missing.append(f"Transformer.{tx_name}: {b}")
                continue

            x, y = bus_coords[b]

            ax.plot(
                [x0, x],
                [y0, y],
                linewidth=3.0,
                zorder=2,
            )

    # ------------------------------------------------------------------
    # 3. Bus dots
    # ------------------------------------------------------------------
    if dots:
        xs = []
        ys = []

        for bus, (x, y) in bus_coords.items():
            xs.append(x)
            ys.append(y)

        ax.scatter(
            xs,
            ys,
            s=18,
            zorder=5,
        )

    # ------------------------------------------------------------------
    # 4. Generator buses
    # ------------------------------------------------------------------
    generator_buses = set()

    for gen_name in dss.Generators.AllNames():
        dss.Generators.Name(gen_name)

        buses = dss.CktElement.BusNames()

        if buses:
            generator_buses.add(bus_name(buses[0]))

    for bus in generator_buses:
        if bus not in bus_coords:
            continue

        x, y = bus_coords[bus]

        ax.scatter(
            [x],
            [y],
            marker="^",
            s=140,
            zorder=10,
            label="Generator",
        )

    # ------------------------------------------------------------------
    # 5. Load buses
    # ------------------------------------------------------------------
    load_buses = set()

    for load_name in dss.Loads.AllNames():
        dss.Loads.Name(load_name)

        buses = dss.CktElement.BusNames()

        if buses:
            load_buses.add(bus_name(buses[0]))

    for bus in load_buses:
        if bus not in bus_coords:
            continue

        x, y = bus_coords[bus]

        ax.scatter(
            [x],
            [y],
            marker="o",
            s=45,
            zorder=8,
            label="Load",
        )

    # ------------------------------------------------------------------
    # 6. Optional bus labels
    # ------------------------------------------------------------------
    if labels:
        for bus, (x, y) in bus_coords.items():
            ax.text(
                x,
                y,
                bus,
                fontsize=7,
                ha="left",
                va="bottom",
                zorder=20,
            )

    # ------------------------------------------------------------------
    # 7. Presentation
    # ------------------------------------------------------------------
    ax.set_title(
        "OpenDSS Distribution Network Circuit Plot "
        "(Matplotlib fallback)",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.15)

    handles, labels_found = ax.get_legend_handles_labels()

    if handles:
        unique = dict(zip(labels_found, handles))
        ax.legend(
            unique.values(),
            unique.keys(),
            loc="upper right",
        )

    if missing:
        print(
            f"Matplotlib fallback: skipped {len(missing)} "
            "elements with missing bus coordinates."
        )

    fig.tight_layout()

    return fig


def validate_network_plot(fig: plt.Figure, save_path: Optional[Union[str, Path]] = None):
    """
    Validates that the generated OpenDSS distribution network plot is not empty.
    Checks axes collections/lines/elements as well as rendered image pixels if saved.
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
    Initializes OpenDSS plant session and generates native OpenDSS circuit plot
    using plant parameters directly from plant.py with proper labels for lines, loads, generator, and transformers.
    Validates that the resulting distribution network plot is not empty, throwing an error if empty.
    """
    from dss import plot as dss_plot

    runner = CoSimulationRunner()
    if dss is None:
        dss = runner.dss

    plant_data = runner.initialize_plant_session(use_baseline_feeder=use_baseline_transformers, seed=42)

    # 1. Assign spatial coordinates dynamically from plant_data topology parameters and OpenDSS model angles
    bus_coords = assign_bus_coordinates_from_plant(dss, plant_data)

    # 2. Enable DSS-Extensions plotting with show=False to preserve Matplotlib figure
    dss_plot.enable(
        plot2d=True,
        plot3d=False,
        show=False,
    )

    # 3. Set Voltagebases and solve so network solution is valid
    dss.run_command("Set Voltagebases=[33.0, 11.0, 0.415]")
    dss.run_command("CalcVoltageBases")
    dss.run_command("solve")

    if mark_transformers:
        dss.run_command("Set MarkTransformers=Y")
    if mark_regulators:
        dss.run_command("Set MarkRegulators=Y")

    dots_str = "Y" if dots else "N"
    # Always set Labels=N to suppress generic node_(x) OpenDSS bus labels
    cmd = f"Plot Circuit Quantity={quantity} Dots={dots_str} Labels=N Subs=N"

    # Native -> Fallback logic
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
                    + len(candidate_ax.texts)
                )

                if total_elements > 0:
                    fig = candidate
                    ax = candidate_ax
                    # Clear any residual raw text node labels from native OpenDSS plot
                    while ax.texts:
                        ax.texts[0].remove()

    except Exception as exc:
        native_plot_error = exc

    if fig is None or ax is None:
        print(
            "Native OpenDSS plotting unavailable or empty; "
            "falling back to direct Matplotlib rendering."
        )

        if native_plot_error is not None:
            print(f"Native plotting error: {native_plot_error}")

        fig = plot_opendss_circuit_matplotlib_fallback(
            dss=dss,
            bus_coords=bus_coords,
            dots=dots,
            labels=labels,
        )

        ax = fig.axes[0]

    # 4. Overlay meaningful network component labels (Generator, Transformers, Representative Loads)
    if "generator_info" not in plant_data or "generator_kw" not in plant_data["generator_info"]:
        raise KeyError("Missing required 'generator_info.generator_kw' in plant_data.")
    gen_kw = plant_data["generator_info"]["generator_kw"]

    gen_buses = set()
    for g in dss.Generators.AllNames():
        dss.Generators.Name(g)
        gen_buses.add(dss.CktElement.BusNames()[0].split(".")[0].lower())

    for gbus in gen_buses:
        if gbus in bus_coords:
            gx, gy = bus_coords[gbus]
            ax.plot(gx, gy, "^", color="darkgreen", markersize=13, label="Generator", zorder=20)
            ax.text(
                gx + 15, gy - 10,
                f"Generator (Source 33kV, {gen_kw/1000.0:.1f}MW)",
                fontsize=9, fontweight="bold", color="darkgreen", zorder=21
            )

    # Substation & Distribution Transformers
    tx_buses = set()
    for t in dss.Transformers.AllNames():
        dss.Transformers.Name(t)
        for b in dss.CktElement.BusNames():
            tx_buses.add(b.split(".")[0].lower())

    # Substation Transformer 33/11kV
    if "sourcebus" in bus_coords:
        sx, sy = bus_coords["sourcebus"]
        ax.plot(sx, sy + 50, "s", color="crimson", markersize=11, label="Transformer", zorder=18)
        ax.text(
            sx + 15, sy + 50,
            "Transformer (Substation 33/11kV)",
            fontsize=8, fontweight="bold", color="crimson", zorder=19
        )

    # Distribution Transformers 11/0.415kV at feeder heads
    for idx, fbus in enumerate(["feeder1_head", "feeder2_head", "feeder3_head"]):
        if fbus in bus_coords:
            tx, ty = bus_coords[fbus]
            ax.plot(tx, ty, "s", color="crimson", markersize=11, zorder=18)
            ax.text(
                tx + 12, ty,
                f"Transformer (Distribution 11/0.415kV Tx{idx+1})",
                fontsize=8, fontweight="bold", color="darkred", zorder=19
            )

    # Consumer Loads
    if "registry" not in plant_data:
        raise KeyError("Missing required 'registry' in plant_data.")
    registry = plant_data["registry"]

    all_load_buses = set()
    for l in dss.Loads.AllNames():
        dss.Loads.Name(l)
        all_load_buses.add(dss.CktElement.BusNames()[0].split(".")[0].lower())

    for lbus in all_load_buses:
        if lbus in bus_coords:
            lx, ly = bus_coords[lbus]
            ax.plot(lx, ly, "o", color="royalblue", markersize=7, label="Loads", zorder=16)

    # Label exactly 1 representative load for each equipment type present in the consumer registry
    labeled_types = set()
    label_count = 0
    for unit in registry.get_all_consumers():
        for ld in unit.loads:
            ltype = ld.load_type
            if ltype not in labeled_types and unit.bus_id.lower() in bus_coords:
                labeled_types.add(ltype)
                label_count += 1
                bx, by = bus_coords[unit.bus_id.lower()]
                formatted_label = ltype.replace("_", " ").title()
                # Offset position slightly based on label count to avoid overlapping text boxes
                dx = 15 if (label_count % 2 == 1) else -130
                dy = ((label_count - 1) % 4) * 18 - 15
                ax.text(
                    bx + dx, by + dy,
                    f"Load: {formatted_label}",
                    fontsize=8, fontweight="bold", color="navy",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", edgecolor="royalblue", alpha=0.85),
                    zorder=22
                )

    # Lines (dummy handle for legend)
    ax.plot([], [], "-", color="black", linewidth=2, label="Distribution Lines")

    # Add clean legend
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

    # 5. Validate plot non-emptiness
    validate_network_plot(fig, save_path=save_path)

    return fig


if __name__ == "__main__":
    plot_opendss_circuit(save_path="src/visualization/dss_circuit_plot.png")
    print("OpenDSS circuit plot generated, validated, and saved successfully.")
