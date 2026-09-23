import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner


def assign_bus_coordinates(dss_instance) -> dict:
    """
    Assigns spatial coordinates (X, Y) to all buses in the active OpenDSS circuit
    so that OpenDSS graphics/plot elements can render the distribution network layout.
    """
    buses = dss_instance.Circuit.AllBusNames()
    coords = {}

    for b in buses:
        b_lower = b.lower()
        if b_lower == "sourcebus":
            c = (0.0, 0.0)
        elif b_lower == "main_bus":
            c = (0.0, 100.0)
        elif b_lower == "feeder1_head":
            c = (-250.0, 250.0)
        elif b_lower == "feeder1_sec":
            c = (-250.0, 300.0)
        elif b_lower == "feeder2_head":
            c = (0.0, 250.0)
        elif b_lower == "feeder2_sec":
            c = (0.0, 300.0)
        elif b_lower == "feeder3_head":
            c = (250.0, 250.0)
        elif b_lower == "feeder3_sec":
            c = (250.0, 300.0)
        elif b_lower.startswith("f1_node"):
            try:
                node_num = int(b_lower.replace("f1_node", ""))
            except ValueError:
                node_num = 1
            dx = ((node_num % 5) - 2) * 45.0
            dy = 300.0 + node_num * 35.0
            c = (-250.0 + dx, dy)
        elif b_lower.startswith("f2_node"):
            try:
                node_num = int(b_lower.replace("f2_node", ""))
            except ValueError:
                node_num = 1
            dx = ((node_num % 5) - 2) * 45.0
            dy = 300.0 + node_num * 35.0
            c = (0.0 + dx, dy)
        elif b_lower.startswith("f3_node"):
            try:
                node_num = int(b_lower.replace("f3_node", ""))
            except ValueError:
                node_num = 1
            dx = ((node_num % 5) - 2) * 45.0
            dy = 300.0 + node_num * 35.0
            c = (250.0 + dx, dy)
        else:
            err_msg = f"Bus '{b}' missing coordinate mapping in assign_bus_coordinates"
            raise ValueError(err_msg)

        coords[b_lower] = c
        dss_instance.run_command(f"SetBusXY Bus={b} x={c[0]} y={c[1]}")

    return coords


from typing import Optional, Union

def validate_network_plot(fig: plt.Figure, save_path: Optional[Union[str, Path]] = None):
    """
    Validates that the generated OpenDSS distribution network plot is not empty.
    Checks axes collections/lines/elements as well as rendered image pixels if saved.
    Raises ValueError or RuntimeError if empty.
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

        # Check non-white/non-transparent pixel content
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
    labels: bool = True,
    subs: bool = True,
    mark_transformers: bool = True,
    mark_regulators: bool = True,
    save_path: str = "src/visualization/dss_circuit_plot.png"
):
    """
    Initializes OpenDSS plant session and generates native OpenDSS circuit plot
    using DSS-Extensions graphics/plot elements with proper labels for lines, loads, generator, and transformers.
    Validates that the resulting distribution network plot is not empty, throwing an error if empty.
    """
    import dss as dss_py

    runner = CoSimulationRunner()
    if dss is None:
        dss = runner.dss

    runner.initialize_plant_session(use_baseline_feeder=use_baseline_transformers, seed=42)

    # 1. Assign spatial coordinates to all circuit buses for OpenDSS plotting
    bus_coords = assign_bus_coordinates(dss)

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
    labels_str = "Y" if labels else "N"
    subs_str = "Y" if subs else "N"

    cmd = f"Plot Circuit Quantity={quantity} Dots={dots_str} Labels={labels_str} Subs={subs_str}"
    dss.run_command(cmd)

    fignums = plt.get_fignums()
    if not fignums:
        raise RuntimeError("OpenDSS plot command failed to produce a Matplotlib figure.")

    fig = plt.figure(fignums[-1])
    ax = fig.axes[0]

    # 4. Overlay specific network element markers and labels (lines, loads, generator, transformers)
    # Generator
    gen_buses = set()
    for g in dss.Generators.AllNames():
        dss.Generators.Name(g)
        gen_buses.add(dss.CktElement.BusNames()[0].split(".")[0].lower())

    for gbus in gen_buses:
        if gbus in bus_coords:
            gx, gy = bus_coords[gbus]
            ax.plot(gx, gy, "^", color="darkgreen", markersize=13, label="Generator", zorder=20)
            ax.text(gx + 15, gy, "Gen (1.5MW)", fontsize=9, fontweight="bold", color="darkgreen", zorder=21)

    # Transformers
    tx_buses = set()
    for t in dss.Transformers.AllNames():
        dss.Transformers.Name(t)
        for b in dss.CktElement.BusNames():
            tx_buses.add(b.split(".")[0].lower())

    for tbus in tx_buses:
        if tbus in bus_coords and tbus in ["sourcebus", "feeder1_head", "feeder2_head", "feeder3_head"]:
            tx, ty = bus_coords[tbus]
            ax.plot(tx, ty, "s", color="crimson", markersize=11, label="Transformer", zorder=18)

    # Loads
    load_buses = set()
    for l in dss.Loads.AllNames():
        dss.Loads.Name(l)
        load_buses.add(dss.CktElement.BusNames()[0].split(".")[0].lower())

    for lbus in load_buses:
        if lbus in bus_coords:
            lx, ly = bus_coords[lbus]
            ax.plot(lx, ly, "o", color="royalblue", markersize=7, label="Loads", zorder=16)

    # Lines (dummy handle for legend representation)
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
