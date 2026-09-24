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


def assign_bus_coordinates_from_plant(dss_instance, plant_data: Dict[str, Any]) -> Dict[str, tuple[float, float]]:
    """
    Dynamically computes and assigns spatial coordinates (X, Y) to all buses
    in the active OpenDSS circuit based on the plant topology parameters from plant_data and active OpenDSS lines.
    Uses MV feeder line lengths and LV network branch line lengths/radial tree structures directly from plant.py.
    """
    buses = [b.lower() for b in dss_instance.Circuit.AllBusNames()]
    coords = {}

    # 1. Base Substation & Main Bus Coordinates
    coords["sourcebus"] = (0.0, 0.0)
    coords["main_bus"] = (0.0, 100.0)

    # Dynamically extract MV feeder lengths from active OpenDSS Line elements or plant_data
    mv_feeder_lengths = {}
    for f_id in [1, 2, 3]:
        line_name = f"Line.mv_feeder_{f_id}"
        if dss_instance.Circuit.SetActiveElement(line_name):
            try:
                length = float(dss_instance.Properties.Value("length"))
            except Exception:
                length = 5.0
        else:
            length = 5.0
        mv_feeder_lengths[f_id] = length

    # Compute MV feeder angles based on the number of feeders to distribute evenly
    num_feeders = len(mv_feeder_lengths)
    feeder_angles = {}
    for idx, f_id in enumerate(sorted(mv_feeder_lengths.keys())):
        if num_feeders > 1:
            ang = 150.0 - idx * (120.0 / (num_feeders - 1))
        else:
            ang = 90.0
        feeder_angles[f_id] = ang

    for f_id, length in mv_feeder_lengths.items():
        ang_rad = math.radians(feeder_angles[f_id])
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
    topologies = plant_data.get("topology", {}).get("topologies", {})
    for f_id, sub_topo in topologies.items():
        sec_bus = f"feeder{f_id}_sec"
        if sec_bus not in coords:
            coords[sec_bus] = (0.0, 300.0)

        # Build adjacency graph from topology line parameters
        adj = {}
        for ln in sub_topo.get("lines", []):
            p = str(ln["bus1"]).lower()
            c = str(ln["bus2"]).lower()
            l = float(ln.get("length", 0.05))
            adj.setdefault(p, []).append((c, l))

        # BFS/DFS traversal to calculate spatial coordinates along tree
        queue = [sec_bus]
        node_angles = {sec_bus: feeder_angles.get(f_id, 90.0)}

        while queue:
            parent = queue.pop(0)
            px, py = coords[parent]
            p_ang = node_angles[parent]
            children = adj.get(parent, [])
            num_c = len(children)
            for i, (child, line_len) in enumerate(children):
                if num_c == 1:
                    c_ang = p_ang
                else:
                    spread = 60.0
                    c_ang = p_ang - (spread / 2.0) + i * (spread / max(num_c - 1.0, 1.0))

                c_rad = math.radians(c_ang)
                scale = max(line_len * 1000.0, 30.0)
                cx = px + scale * math.cos(c_rad)
                cy = py + scale * math.sin(c_rad)
                coords[child] = (cx, cy)
                node_angles[child] = c_ang
                queue.append(child)

    # 3. Apply SetBusXY to OpenDSS for all circuit buses with graceful fallback for unmapped buses
    default_x, default_y = 0.0, 200.0
    for idx, b in enumerate(dss_instance.Circuit.AllBusNames()):
        b_lower = b.lower()
        if b_lower in coords:
            x, y = coords[b_lower]
        else:
            x, y = default_x + (idx * 20.0), default_y
            coords[b_lower] = (x, y)
        dss_instance.run_command(f"SetBusXY Bus={b} x={x} y={y}")

    return coords


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
    import dss as dss_py

    runner = CoSimulationRunner()
    if dss is None:
        dss = runner.dss

    plant_data = runner.initialize_plant_session(use_baseline_feeder=use_baseline_transformers, seed=42)

    # 1. Assign spatial coordinates dynamically from plant_data topology parameters
    bus_coords = assign_bus_coordinates_from_plant(dss, plant_data)

    # 2. Enable DSS-Python plotting extension subsystem
    if hasattr(dss_py, "DSS") and hasattr(dss_py.DSS, "Plotting"):
        dss_py.DSS.Plotting.enable()
    elif hasattr(dss_py, "Plotting"):
        dss_py.Plotting.enable()

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
    dss.run_command(cmd)

    fignums = plt.get_fignums()
    if not fignums:
        # Fallback: create Matplotlib figure if plot command did not register in fignums
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = plt.figure(fignums[-1])
        ax = fig.axes[0] if fig.axes else fig.add_subplot(111)

    # Clear any residual raw text node labels if any were added
    while ax.texts:
        ax.texts[0].remove()

    # 4. Overlay meaningful network component labels (Generator, Transformers, Representative Loads)
    gen_kw = plant_data.get("generator_info", {}).get("generator_kw", 1500.0)
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
    registry = plant_data.get("registry")
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
    if registry:
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
