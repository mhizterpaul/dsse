import os
import sys
import math
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pypsa
import networkx as nx
import schemdraw
import schemdraw.elements as elm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner


def normalise_bus_name(raw_name: str) -> str:
    """Map a terminal name such as 'bus1.1.2.3' to its bus identifier."""
    return str(raw_name).split(".", 1)[0].strip().lower()


def read_min_phase_voltage_pu(dss) -> pd.Series:
    """
    Read the minimum available phase-voltage magnitude in per-unit
    from the existing solved OpenDSS circuit.
    """
    dss.run_command("Set Voltagebases=[33.0, 11.0, 0.415]")
    dss.run_command("CalcVoltageBases")
    dss.run_command("solve")

    if not dss.Solution.Converged():
        raise RuntimeError(
            "OpenDSS has no converged solution. Solve the circuit in the simulation layer before plotting."
        )

    values = {}

    for raw_bus in dss.Circuit.AllBusNames():
        b_idx = dss.Circuit.SetActiveBus(raw_bus)
        if b_idx is None or b_idx < 0:
            raise ValueError(f"Cannot activate OpenDSS bus {raw_bus!r}.")

        kv_base = float(dss.Bus.kVBase())
        if not math.isfinite(kv_base) or kv_base <= 0:
            raise ValueError(
                f"Bus {raw_bus!r} has no valid voltage base: {kv_base}"
            )

        nodes = np.asarray(dss.Bus.Nodes(), dtype=int)
        pairs = np.asarray(
            dss.Bus.puVmagAngle(),
            dtype=float,
        )

        if pairs.size % 2 != 0:
            raise ValueError(
                f"Invalid per-unit voltage magnitude-angle array at {raw_bus!r}."
            )

        pairs = pairs.reshape(-1, 2)

        if len(nodes) != len(pairs):
            raise ValueError(
                f"Node/voltage count mismatch at {raw_bus!r}: "
                f"{len(nodes)} nodes and {len(pairs)} voltage pairs."
            )

        phase_magnitudes = [
            float(mag)
            for node, (mag, _angle) in zip(nodes, pairs)
            if node != 0 and math.isfinite(float(mag)) and mag > 0
        ]

        if phase_magnitudes:
            values[normalise_bus_name(raw_bus)] = min(phase_magnitudes)

    return pd.Series(values, name="min_phase_voltage_pu", dtype=float)


def topology_positions(net: pypsa.Network, root: str = "sourcebus") -> Dict[str, Tuple[float, float]]:
    """Assign stable, feeder-oriented level-based positions to a distribution network."""
    graph = nx.Graph()
    graph.add_nodes_from(net.buses.index)

    for component_name in ("lines", "transformers"):
        table = getattr(net, component_name)
        if table is not None and not table.empty:
            graph.add_edges_from(table[["bus0", "bus1"]].itertuples(index=False, name=None))

    if root not in graph:
        raise KeyError(f"Source bus {root!r} is missing.")

    # Build a deterministic breadth-first spanning tree.
    tree = nx.DiGraph()
    tree.add_node(root)
    queue = deque([root])
    visited = {root}
    level = {root: 0}
    nodes_at_level = {0: [root]}

    while queue:
        curr = queue.popleft()
        curr_lvl = level[curr]
        for neighbor in sorted(graph.neighbors(curr)):
            if neighbor not in visited:
                visited.add(neighbor)
                level[neighbor] = curr_lvl + 1
                nodes_at_level.setdefault(curr_lvl + 1, []).append(neighbor)
                tree.add_edge(curr, neighbor)
                queue.append(neighbor)

    positions = {}
    for lvl, n_list in nodes_at_level.items():
        n_count = len(n_list)
        y = -float(lvl) * 3.0
        for idx, node in enumerate(n_list):
            x = (idx - (n_count - 1) / 2.0) * 2.5
            positions[node] = (x, y)

    for node in net.buses.index:
        if node not in positions:
            positions[node] = (0.0, 0.0)

    return positions


def draw_busbar(
    ax,
    x_positions: List[float],
    y: float,
    padding: float = 0.25,
    linewidth: float = 2.6,
) -> None:
    """Draw a continuous horizontal busbar."""
    if not x_positions:
        return
    x_left = min(x_positions) - padding
    x_right = max(x_positions) + padding
    if x_right <= x_left:
        x_left -= 0.25
        x_right += 0.25
    ax.plot(
        [x_left, x_right],
        [y, y],
        color="black",
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=5,
    )


def draw_orthogonal_edge(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = "#333333",
    linewidth: float = 1.4,
) -> None:
    """Draw an edge using only horizontal and vertical right-angle segments."""
    x0, y0 = start
    x1, y1 = end
    y_mid = (y0 + y1) / 2.0
    ax.plot(
        [x0, x0, x1, x1],
        [y0, y_mid, y_mid, y1],
        color=color,
        linewidth=linewidth,
        zorder=2,
    )


def draw_ground_signal(
    d: schemdraw.Drawing,
    x: float,
    y: float,
    label: Optional[str] = None,
) -> None:
    """Place a signal-ground symbol (elm.GroundSignal) at designated reference terminal."""
    symbol = elm.GroundSignal(lead=True).at((x, y))
    if label:
        symbol = symbol.label(label, loc="bottom", fontsize=6)
    d.add(symbol)


def draw_consumer_branch(
    d: schemdraw.Drawing,
    ax,
    cx: float,
    busbar_y: float,
    cid: str,
) -> None:
    """Draw a single consumer connection branch perpendicular to the shared busbar."""
    drop_y = busbar_y - 0.7

    # Vertical drop line from busbar
    ax.plot([cx, cx], [busbar_y, drop_y], color="#333333", linewidth=1.2, zorder=4)

    # Load Resistor symbol
    d.add(
        elm.Resistor()
        .at((cx, drop_y))
        .down()
        .label(cid, loc="bottom", fontsize=6, color="navy")
    )


def build_pypsa_network_from_opendss(dss, plant_data: Dict[str, Any]) -> pypsa.Network:
    """
    Constructs a PyPSA Network instance populated dynamically from active OpenDSS circuit elements,
    plant_data topology parameters, and consumer units from registry without synthetic hubs or fallbacks.
    Raises KeyError or ValueError if required network elements or properties are missing.
    """
    dss.run_command("Set Voltagebases=[33.0, 11.0, 0.415]")
    dss.run_command("CalcVoltageBases")

    net = pypsa.Network()

    if "AC" not in net.carriers.index:
        net.add("Carrier", "AC", color="steelblue", nice_name="AC electricity")
    if "grid" not in net.carriers.index:
        net.add("Carrier", "grid", color="slategray", nice_name="Grid supply")

    def clean_bus_name(raw_bus: str) -> str:
        return raw_bus.split(".")[0].strip().lower()

    # 1. Add Buses with kVBase as v_nom
    for bus_name in dss.Circuit.AllBusNames():
        b_clean = clean_bus_name(bus_name)

        b_idx = dss.Circuit.SetActiveBus(bus_name)
        if b_idx is None or b_idx < 0:
            raise ValueError(f"Cannot activate OpenDSS bus {bus_name!r}.")

        kv_base = float(dss.Bus.kVBase())
        if not math.isfinite(kv_base) or kv_base <= 0:
            err_msg = f"Invalid base voltage evaluated for bus '{bus_name}': {kv_base} kV"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

        v_nom = kv_base * (3.0 ** 0.5) if kv_base < 25.0 else kv_base
        net.add("Bus", b_clean, carrier="AC", v_nom=v_nom)

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

    net.add(
        "Generator",
        "gen_33kv",
        bus="sourcebus",
        carrier="grid",
        p_nom=gen_kw / 1000.0,
        control="Slack"
    )

    # 3. Add Lines with physical properties queried directly from OpenDSS active elements
    for line_name in dss.Lines.AllNames():
        if not dss.Circuit.SetActiveElement(f"Line.{line_name}"):
            continue

        buses = [clean_bus_name(b) for b in dss.CktElement.BusNames()]
        if len(buses) < 2:
            raise ValueError(f"Line '{line_name}' has fewer than 2 terminal buses: {buses}")

        b1, b2 = buses[0], buses[1]
        if b1 not in net.buses.index or b2 not in net.buses.index:
            raise KeyError(f"Line '{line_name}' references missing bus endpoints: {b1}, {b2}")

        length = float(dss.Properties.Value("length"))
        r1 = float(dss.Properties.Value("r1"))
        x1 = float(dss.Properties.Value("x1"))

        r_ohm = r1 * length
        x_ohm = x1 * length

        if length <= 0 or r_ohm < 0:
            err_msg = f"Invalid line parameters for '{line_name}': length={length}, r={r_ohm}"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

        net.add("Line", line_name, bus0=b1, bus1=b2, r=r_ohm, x=x_ohm)

    # 4. Add Transformers with ratings and impedances queried directly from OpenDSS properties
    for tx_name in dss.Transformers.AllNames():
        if not dss.Circuit.SetActiveElement(f"Transformer.{tx_name}"):
            continue

        buses = [clean_bus_name(b) for b in dss.CktElement.BusNames()]
        if len(buses) < 2:
            raise ValueError(f"Transformer '{tx_name}' has fewer than 2 terminal buses: {buses}")

        b1, b2 = buses[0], buses[1]
        if b1 not in net.buses.index or b2 not in net.buses.index:
            raise KeyError(f"Transformer '{tx_name}' references missing bus endpoints: {b1}, {b2}")

        mva_val = float(dss.Properties.Value("kVA")) / 1000.0
        if mva_val <= 0:
            err_msg = f"Invalid transformer capacity rating for '{tx_name}': {mva_val} MVA"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

        r_pu = 0.01
        x_pu = 0.05
        try:
            r_val = float(dss.Properties.Value("%R"))
            if r_val > 0:
                r_pu = r_val / 100.0
        except Exception:
            pass

        try:
            x_val = float(dss.Properties.Value("XHL"))
            if x_val > 0:
                x_pu = x_val / 100.0
        except Exception:
            pass

        net.add("Transformer", tx_name, bus0=b1, bus1=b2, s_nom=mva_val, r=r_pu, x=x_pu)

    # 5. Add Consumer Loads with kW and kvar demands queried directly from OpenDSS active loads
    for load_name in dss.Loads.AllNames():
        if not dss.Circuit.SetActiveElement(f"Load.{load_name}"):
            continue

        b_load = clean_bus_name(dss.CktElement.BusNames()[0])
        if b_load not in net.buses.index:
            raise KeyError(f"Load '{load_name}' references missing bus '{b_load}'")

        kw = float(dss.Properties.Value("kW"))
        kvar = 0.0
        try:
            kvar = float(dss.Properties.Value("kvar"))
        except Exception:
            pass

        if kw < 0:
            err_msg = f"Invalid negative power demand for load '{load_name}': {kw} kW"
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

        net.add("Load", load_name, bus=b_load, p_set=kw/1000.0, q_set=kvar/1000.0)

    return net


def validate_network_plot(net: pypsa.Network, plant_data: dict, fig: plt.Figure, save_path: Optional[Union[str, Path]] = None):
    """
    Validates structural topology coverage (buses, branches, consumer units) and graphical non-emptiness.
    Raises ValueError if validation fails.
    """
    if not fig or not fig.axes:
        raise ValueError("Generated distribution network plot figure or axes is empty.")

    ax = fig.axes[0]
    total_elements = len(ax.collections) + len(ax.lines) + len(ax.patches) + len(ax.texts)
    if total_elements == 0:
        raise ValueError("Distribution network plot contains no graphical elements or network lines.")

    # Structural Coverage Validations
    if len(net.buses) == 0:
        raise ValueError("PyPSA network contains no buses.")

    if len(net.lines) == 0:
        raise ValueError("PyPSA network contains no transmission/distribution lines.")

    if "registry" in plant_data:
        registry = plant_data["registry"]
        consumers = registry.get_all_consumers()
        if len(consumers) == 0:
            raise ValueError("Consumer registry contains no registered consumer units.")

        unmapped_consumers = [u.consumer_id for u in consumers if u.bus_id.lower() not in net.buses.index]
        if unmapped_consumers:
            raise ValueError(f"Consumer units reference buses missing from PyPSA network: {unmapped_consumers}")

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
    Initializes OpenDSS plant session, reads solved operating state voltages,
    builds PyPSA Network representation, and renders an orthogonal single-line diagram.
    Uses horizontal busbars, right-angle connections, Schemdraw symbols, and signal grounding.
    """
    runner = CoSimulationRunner()
    plant_data = runner.initialize_plant_session(use_baseline_feeder=use_baseline_transformers, seed=42)

    if dss is None:
        dss = runner.dss

    # 1. Read solved minimum per-unit phase voltages from OpenDSS
    voltage_pu = read_min_phase_voltage_pu(dss)

    # 2. Build PyPSA Network instance directly from active OpenDSS session and plant_data
    net = build_pypsa_network_from_opendss(dss, plant_data)

    positions = topology_positions(net, root="sourcebus")

    fig, ax = plt.subplots(figsize=(18, 12))
    d = schemdraw.Drawing(canvas=ax)

    # 3. Source & Root Busbar at sourcebus
    if "sourcebus" in positions:
        sx, sy = positions["sourcebus"]
        d.add(elm.SourceSin().at((sx, sy)).up().label("33 kV Grid Feed-in", loc="top", fontsize=8))
        draw_busbar(ax, [sx - 0.8, sx + 0.8], sy, padding=0.3, linewidth=3.0)

    # 4. Draw Feeder Lines with Right-Angle Orthogonal Routing
    for line_idx, row in net.lines.iterrows():
        b0, b1 = row.bus0, row.bus1
        if b0 in positions and b1 in positions:
            p0 = positions[b0]
            p1 = positions[b1]
            draw_orthogonal_edge(ax, p0, p1, color="#333333", linewidth=1.4)

    # 5. Draw Transformers using Schemdraw
    for tx_idx, row in net.transformers.iterrows():
        b0, b1 = row.bus0, row.bus1
        if b0 in positions and b1 in positions:
            x0, y0 = positions[b0]
            x1, y1 = positions[b1]
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            d.add(
                elm.Transformer()
                .at((mx, my))
                .right()
                .label(f"Trxr {tx_idx}", loc="top", fontsize=7)
            )

    # 6. Draw Bus Annotations (No scattered navy dots)
    for b_name, (x, y) in positions.items():
        v_pu = voltage_pu.get(b_name, 1.0)
        ax.text(
            x + 0.15, y + 0.1,
            f"{b_name} ({v_pu:.3f} p.u.)",
            fontsize=6.5,
            fontweight="bold",
            color="darkgreen" if 0.95 <= v_pu <= 1.05 else "darkred",
            zorder=8
        )

    # 7. Group Consumer Units by Feeder Section / Shared Bus and Render Busbars + Drop Branches
    if "registry" not in plant_data:
        err_msg = "Missing required 'registry' in plant_data"
        print(f"ERROR: {err_msg}")
        raise KeyError(err_msg)

    registry = plant_data["registry"]
    consumers_by_group = {}
    for unit in registry.get_all_consumers():
        b_name = normalise_bus_name(unit.bus_id)
        if b_name not in positions:
            raise KeyError(
                f"Consumer unit '{unit.consumer_id}' at bus '{b_name}' is not represented in the PyPSA network."
            )
        group_key = (unit.feeder_id, b_name)
        consumers_by_group.setdefault(group_key, []).append((b_name, unit.consumer_id))

    for (fid, b_name), consumer_entries in consumers_by_group.items():
        x, y = positions[b_name]
        c_ids = [c_id for _, c_id in consumer_entries]
        count = len(c_ids)

        max_id_len = max(len(cid) for cid in c_ids) if c_ids else 10
        pitch = max(0.8, max_id_len * 0.04)
        branch_x_list = [x + (i - (count - 1) / 2.0) * pitch for i in range(count)]

        # Continuous shared PCC busbar across consumer group
        busbar_y = y - 0.2
        draw_busbar(ax, branch_x_list, busbar_y, padding=0.25, linewidth=2.6)

        # Neutral ground signal plotted strictly at PCC busbars where more than one consumer units are connected
        if count > 1:
            pcc_ground_x = max(branch_x_list) + 0.25
            drop_y = busbar_y - 0.7
            ax.plot([pcc_ground_x, pcc_ground_x], [busbar_y, drop_y], color="#333333", linewidth=1.2, zorder=4)
            draw_ground_signal(d, pcc_ground_x, drop_y)

        # Draw individual perpendicular connection drops for each consumer
        for cx, cid in zip(branch_x_list, c_ids):
            draw_consumer_branch(d, ax, cx, busbar_y, cid)

    d.draw()

    ax.set_title("OpenDSS Distribution Network — Orthogonal Schematic Single-Line Diagram", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.15)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    # Save plot
    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, bbox_inches="tight", dpi=300)

    # Validate topology and non-emptiness
    validate_network_plot(net, plant_data, fig, save_path=save_path)

    return fig


if __name__ == "__main__":
    plot_opendss_circuit(save_path="src/visualization/dss_circuit_plot.png")
    print("OpenDSS PyPSA distribution network single line diagram generated, validated, and saved successfully.")
