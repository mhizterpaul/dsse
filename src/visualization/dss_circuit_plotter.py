import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dss import dss
from src.simulation.runner import CoSimulationRunner


def plot_opendss_circuit(
    use_baseline_transformers: bool = True,
    quantity: str = "Power",
    dots: bool = True,
    labels: bool = True,
    subs: bool = True,
    mark_transformers: bool = True,
    mark_regulators: bool = True,
    save_path: str = "dss_circuit_plot.png"
):
    """
    Initializes OpenDSS plant session and generates native OpenDSS circuit plot
    using DSS-Extensions plotting subsystem.
    """
    runner = CoSimulationRunner()
    runner.initialize_plant_session(use_baseline_transformers=use_baseline_transformers, seed=42)

    dss.Plotting.enable()

    if mark_transformers:
        dss.Text.Command = "Set MarkTransformers=Y"
    if mark_regulators:
        dss.Text.Command = "Set MarkRegulators=Y"

    dots_str = "Y" if dots else "N"
    labels_str = "Y" if labels else "N"
    subs_str = "Y" if subs else "N"

    cmd = f"Plot Circuit Quantity={quantity} Dots={dots_str} Labels={labels_str} Subs={subs_str}"
    dss.Text.Command = cmd

    fignums = plt.get_fignums()
    if fignums:
        fig = plt.figure(fignums[-1])
        if save_path:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_p, bbox_inches="tight", dpi=300)
        return fig
    return None


if __name__ == "__main__":
    plot_opendss_circuit(save_path="src/visualization/dss_circuit_plot.png")
    print("OpenDSS circuit plot generated and saved successfully.")
