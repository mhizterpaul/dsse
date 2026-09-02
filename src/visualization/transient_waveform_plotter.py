import os
import sys
import traceback
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner
from src.transient.events import SingleEquipmentSwitchEvent
from src.simulation.filter import remove_low_frequency_components


def simulate_and_plot_equipment_group(group_id: int = 1):
    """
    Simulates and plots 3-phase transient waveforms for a specific equipment group
    (group_id=1 for Equipment Group 1, group_id=2 for Equipment Group 2) evaluated from ATP
    under OpenDSS steady-state parameters. Creates and returns only a single matplotlib figure.
    """
    if group_id == 1:
        equipment_types = ["ac_motor", "dc_motor_inverter", "microwave", "induction_plate"]
    elif group_id == 2:
        equipment_types = ["compressor", "audio_amplifier", "ups", "industrial_fan"]
    else:
        raise ValueError(f"Invalid equipment group_id: {group_id}. Expected 1 or 2.")

    titles = {
        "ac_motor": "AC Motor Switch Transient",
        "dc_motor_inverter": "DC Motor Inverter Switch Transient",
        "microwave": "Microwave Switch Transient",
        "induction_plate": "Induction Plate Switch Transient",
        "compressor": "Compressor Switch Transient",
        "audio_amplifier": "Audio Amplifier Switch Transient",
        "ups": "UPS Switch Transient",
        "industrial_fan": "Industrial Fan Switch Transient"
    }

    runner = CoSimulationRunner()
    runner.initialize_plant_session(use_baseline_transformers=True, seed=42)

    waveforms = {}

    from src.power_plant import plant

    for eq in equipment_types:
        s_ev = SingleEquipmentSwitchEvent(eq, 0.02, 0.04, "feeder1_head", {})
        t, v_dict, i_dict, meta = runner.measure_transients(
            op=plant.solve_operating_point(runner.dss),
            event=s_ev,
            scenario_id=f"vis_g{group_id}_{eq}",
            feeder_idx=1
        )

        v_raw = v_dict["trans1"]
        i_raw = i_dict["trans1"]

        v_high = remove_low_frequency_components(v_raw)
        i_high = remove_low_frequency_components(i_raw)

        waveforms[eq] = {"time": t, "voltage": v_high, "current": i_high}

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for idx, eq in enumerate(equipment_types):
        ax = axes[idx]
        data = waveforms[eq]
        ax.plot(data["time"] * 1000.0, data["current"][:, 0], 'r-', label='Phase A Current')
        ax.plot(data["time"] * 1000.0, data["current"][:, 1], 'g-', label='Phase B Current')
        ax.plot(data["time"] * 1000.0, data["current"][:, 2], 'b-', label='Phase C Current')
        ax.set_title(titles[eq], fontsize=11, fontweight='bold')
        ax.set_ylabel("Current (A)")
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()

    return fig


def simulate_and_plot_load_circuit_transients():
    """
    Convenience wrapper returning figures for both equipment groups.
    """
    fig1 = simulate_and_plot_equipment_group(1)
    fig2 = simulate_and_plot_equipment_group(2)
    return fig1, fig2


if __name__ == "__main__":
    fig1, fig2 = simulate_and_plot_load_circuit_transients()
    plt.close("all")
    print("Simulated and generated transient figures successfully.")
