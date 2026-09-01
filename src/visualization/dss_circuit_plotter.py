import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.runner import CoSimulationRunner


def plot_opendss_circuit(
    dss=None,
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
    using DSS-Extensions plotting subsystem. Passed single dss instance from runner.
    """
    runner = CoSimulationRunner()
    if dss is None:
        dss = runner.dss
    runner.initialize_plant_session(use_baseline_transformers=use_baseline_transformers, seed=42)

    # Set Voltagebases and solve so network solution is valid
    dss.run_command("Set Voltagebases=[33.0, 11.0, 0.415]")
    dss.run_command("CalcVoltageBases")
    dss.run_command("solve")

    dss_py = dss.to_dss_python() if hasattr(dss, "to_dss_python") else dss
    if hasattr(dss_py, "Plotting"):
        dss_py.Plotting.enable()

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
