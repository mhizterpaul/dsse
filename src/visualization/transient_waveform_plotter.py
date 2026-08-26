import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner
from src.transient.events import SingleEquipmentSwitchEvent
from src.simulation.filter import remove_low_frequency_components


def simulate_and_plot_load_circuit_transients():
    """
    Simulates and plots 3-phase transient waveforms for 8 consumer load circuit types
    evaluated from ATP under OpenDSS steady-state parameters.
    Splits the 8 load circuits into 2 figures (4 load circuits each).
    """
    equipment_types = [
        "ac_motor", "dc_motor_inverter", "microwave", "induction_plate",
        "compressor", "audio_amplifier", "ups", "industrial_fan"
    ]

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

    for eq in equipment_types:
        s_ev = SingleEquipmentSwitchEvent(eq, 0.02, 0.04, "feeder1_head", {})
        sim_res = runner.run_simulation(
            events=[s_ev],
            use_baseline_transformers=True,
            include_load_event=True,
            include_fault_event=False,
            scenario_id=f"vis_{eq}",
            seed=42,
            reinitialize_plant=False
        )

        m_id = "trans1_lv_boundary_consumer_unit"
        cu = sim_res.processed_consumer_units.get(m_id)

        t = sim_res.time_s if (sim_res.time_s is not None and len(sim_res.time_s) > 0) else np.linspace(0.0, 0.1, 1000)

        if cu is not None:
            v_high = remove_low_frequency_components(cu["raw_voltage"])
            i_high = remove_low_frequency_components(cu["raw_current"])
        else:
            v_high = np.zeros((len(t), 3))
            i_high = np.zeros((len(t), 3))

        waveforms[eq] = {"time": t, "voltage": v_high, "current": i_high}

    # Group 1: First 4 equipment types
    fig1, axes1 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for idx, eq in enumerate(equipment_types[:4]):
        ax = axes1[idx]
        data = waveforms[eq]
        ax.plot(data["time"] * 1000.0, data["current"][:, 0], 'r-', label='Phase A Current')
        ax.plot(data["time"] * 1000.0, data["current"][:, 1], 'g-', label='Phase B Current')
        ax.plot(data["time"] * 1000.0, data["current"][:, 2], 'b-', label='Phase C Current')
        ax.set_title(titles[eq], fontsize=11, fontweight='bold')
        ax.set_ylabel("Current (A)")
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right")
    axes1[-1].set_xlabel("Time (ms)")
    fig1.tight_layout()

    # Group 2: Next 4 equipment types
    fig2, axes2 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for idx, eq in enumerate(equipment_types[4:]):
        ax = axes2[idx]
        data = waveforms[eq]
        ax.plot(data["time"] * 1000.0, data["current"][:, 0], 'r-', label='Phase A Current')
        ax.plot(data["time"] * 1000.0, data["current"][:, 1], 'g-', label='Phase B Current')
        ax.plot(data["time"] * 1000.0, data["current"][:, 2], 'b-', label='Phase C Current')
        ax.set_title(titles[eq], fontsize=11, fontweight='bold')
        ax.set_ylabel("Current (A)")
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right")
    axes2[-1].set_xlabel("Time (ms)")
    fig2.tight_layout()

    return fig1, fig2


if __name__ == "__main__":
    fig1, fig2 = simulate_and_plot_load_circuit_transients()
    plt.close("all")
    print("Simulated and generated transient figures successfully.")
