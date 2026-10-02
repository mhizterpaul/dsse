import json

with open("report.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Original setup cell is nb['cells'][1]
# Cells 0..4 are setup and Stage 1 datasets/OpenDSS plot
stage1_cells = nb["cells"][:5]

# Add schemdraw to setup cell imports if needed
setup_src = nb["cells"][1]["source"]

# Define 3 dedicated Stage 2 clean experiment cells
stage2_md = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Stage 2: Clean Experiment — Single No-Load Baseline, 1 Equipment Load, and 1 Binary Load Pair\n",
    "This stage executes the clean experiment on 1 equipment (AC Motor) and 1 binary load pair (AC Motor + Compressor). A single no-load baseline solution $x_{\\mathrm{no-load}}(t)$ is obtained ONCE from ATP and subtracted from all loaded conditions."
   ]
}

cell_1_md = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cell 1: ATP Case Generation, Three-Phase Overlay Plots, and Single No-Load Subtraction\n",
    "Runs the clean experiment simulation suite ONCE to obtain the single $x_{\\mathrm{no-load}}(t)$ baseline solution alongside loaded conditions for Single Load 1 (AC Motor) and Joint Pair (AC Motor + Compressor). Plots three-phase overlays and load-induced residuals."
   ]
}

cell_1_code = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from src.transient.no_load_decomposition import run_clean_experiment_simulations\n",
    "\n",
    "# Execute clean experiment simulation suite ONCE (1 equipment load: AC Motor, 1 binary pair: AC Motor + Compressor)\n",
    "sim_data = run_clean_experiment_simulations(\"ac_motor\", \"compressor\", start_time_s=0.02, duration_s=0.10)\n",
    "t = sim_data[\"time\"]\n",
    "v_no = sim_data[\"v_no_load\"]\n",
    "v_s1 = sim_data[\"v_single_1\"]\n",
    "v_pair = sim_data[\"v_pair\"]\n",
    "i_no = sim_data[\"i_no_load\"]\n",
    "i_s1 = sim_data[\"i_single_1\"]\n",
    "i_pair = sim_data[\"i_pair\"]\n",
    "\n",
    "phases = [\"Va\", \"Vb\", \"Vc\"]\n",
    "colors = [\"crimson\", \"darkgreen\", \"navy\"]\n",
    "\n",
    "# 1. Three-Phase Voltage Overlay (No-Load vs Single Load 1 vs Joint Pair)\n",
    "fig_v, ax_v = plt.subplots(figsize=(14, 5))\n",
    "for i, p in enumerate(phases):\n",
    "    ax_v.plot(t, v_no[:, i], color=colors[i], label=f\"{p} no-load\", alpha=0.5, linestyle=\":\")\n",
    "    ax_v.plot(t, v_s1[:, i], color=colors[i], label=f\"{p} single load 1\", alpha=0.8, linestyle=\"--\")\n",
    "    ax_v.plot(t, v_pair[:, i], color=colors[i], label=f\"{p} joint pair\", linewidth=1.8, linestyle=\"-\")\n",
    "ax_v.set_xlabel(\"Time (s)\", fontweight=\"bold\")\n",
    "ax_v.set_ylabel(\"Voltage (V)\", fontweight=\"bold\")\n",
    "ax_v.set_title(\"Three-Phase Transformer Voltage: Single No-Load Baseline vs Loaded Conditions\", fontweight=\"bold\")\n",
    "ax_v.legend(ncol=3, loc=\"upper right\", frameon=True, fontsize=8)\n",
    "ax_v.grid(True, linestyle=\":\", alpha=0.6)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# 2. Three-Phase Current Overlay\n",
    "fig_i, ax_i = plt.subplots(figsize=(14, 5))\n",
    "for i, p in enumerate([\"Ia\", \"Ib\", \"Ic\"]):\n",
    "    ax_i.plot(t, i_no[:, i], color=colors[i], label=f\"{p} no-load\", alpha=0.5, linestyle=\":\")\n",
    "    ax_i.plot(t, i_s1[:, i], color=colors[i], label=f\"{p} single load 1\", alpha=0.8, linestyle=\"--\")\n",
    "    ax_i.plot(t, i_pair[:, i], color=colors[i], label=f\"{p} joint pair\", linewidth=1.8, linestyle=\"-\")\n",
    "ax_i.set_xlabel(\"Time (s)\", fontweight=\"bold\")\n",
    "ax_i.set_ylabel(\"Current (A)\", fontweight=\"bold\")\n",
    "ax_i.set_title(\"Three-Phase Transformer Current: Single No-Load Baseline vs Loaded Conditions\", fontweight=\"bold\")\n",
    "ax_i.legend(ncol=3, loc=\"upper right\", frameon=True, fontsize=8)\n",
    "ax_i.grid(True, linestyle=\":\", alpha=0.6)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# 3. Subtraction of Single No-Load Solution (Residual Voltage Comparison)\n",
    "v_res_1 = sim_data[\"v_res_1\"]\n",
    "v_res_pair = sim_data[\"v_res_pair\"]\n",
    "\n",
    "fig_res, ax_res = plt.subplots(figsize=(14, 5))\n",
    "ax_res.plot(t, v_res_1[:, 0], \"--\", color=\"navy\", label=\"V_a Single Load 1 Residual (S1 - NoLoad)\", linewidth=1.8)\n",
    "ax_res.plot(t, v_res_pair[:, 0], \"-\", color=\"crimson\", label=\"V_a Joint Load Pair Residual (Pair - NoLoad)\", linewidth=2.0)\n",
    "ax_res.axhline(0, color=\"black\", linewidth=0.8, linestyle=\"--\")\n",
    "ax_res.set_xlabel(\"Time (s)\", fontweight=\"bold\")\n",
    "ax_res.set_ylabel(\"\\\\Delta V (V)\", fontweight=\"bold\")\n",
    "ax_res.set_title(\"Load-Induced Residual Voltage Waveforms (Phase A Voltage)\", fontweight=\"bold\")\n",
    "ax_res.legend(loc=\"upper right\", frameon=True)\n",
    "ax_res.grid(True, linestyle=\":\", alpha=0.6)\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
}

cell_2_md = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cell 2: Time-Frequency Analysis (STFT Spectrogram Comparison)\n",
    "Computes and displays STFT time-frequency spectrograms comparing Single Load 1 vs Joint Load Pair."
   ]
}

cell_2_code = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from src.transient.no_load_decomposition import analyze_stft_spectrum\n",
    "\n",
    "f_stft_1, t_stft_1, Z_1 = analyze_stft_spectrum(t, v_res_1[:, 0])\n",
    "f_stft_p, t_stft_p, Z_p = analyze_stft_spectrum(t, v_res_pair[:, 0])\n",
    "\n",
    "fig_stft, (ax_s1, ax_sp) = plt.subplots(1, 2, figsize=(16, 5))\n",
    "pcm1 = ax_s1.pcolormesh(t_stft_1, f_stft_1, Z_1, shading=\"gouraud\", cmap=\"inferno\")\n",
    "fig_stft.colorbar(pcm1, ax=ax_s1, label=\"Magnitude (V)\")\n",
    "ax_s1.set_xlabel(\"Time (s)\", fontweight=\"bold\")\n",
    "ax_s1.set_ylabel(\"Frequency (Hz)\", fontweight=\"bold\")\n",
    "ax_s1.set_title(\"STFT Spectrogram: Single Load 1 Residual V_a\", fontweight=\"bold\")\n",
    "ax_s1.set_ylim(0, 1000)\n",
    "\n",
    "pcmp = ax_sp.pcolormesh(t_stft_p, f_stft_p, Z_p, shading=\"gouraud\", cmap=\"inferno\")\n",
    "fig_stft.colorbar(pcmp, ax=ax_sp, label=\"Magnitude (V)\")\n",
    "ax_sp.set_xlabel(\"Time (s)\", fontweight=\"bold\")\n",
    "ax_sp.set_ylabel(\"Frequency (Hz)\", fontweight=\"bold\")\n",
    "ax_sp.set_title(\"STFT Spectrogram: Joint Load Pair Residual V_a\", fontweight=\"bold\")\n",
    "ax_sp.set_ylim(0, 1000)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
}

cell_3_md = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cell 3: Waveform Decomposition ($x_{\\mathrm{SSD}} + x_{\\mathrm{harmonic}} + x_{\\mathrm{transient}}$), Derived Wave Equations, and Component Plotting\n",
    "Decomposes residual signals into slowly varying ($x_{\\mathrm{SSD}}$), harmonic ($x_{\\mathrm{harmonic}}$), and transient ($x_{\\mathrm{transient}}$) components, derives explicit mathematical wave equations, and plots components on a common time axis."
   ]
}

cell_3_code = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from IPython.display import display, HTML\n",
    "from src.transient.no_load_decomposition import decompose_load_waveform, derive_wave_equations\n",
    "\n",
    "decomp_s1 = decompose_load_waveform(t, v_res_1[:, 0])\n",
    "decomp_pair = decompose_load_waveform(t, v_res_pair[:, 0])\n",
    "\n",
    "eqs_s1 = derive_wave_equations(t, decomp_s1, signal_name=\"V_{a, single1}\", unit=\"V\")\n",
    "eqs_pair = derive_wave_equations(t, decomp_pair, signal_name=\"V_{a, pair}\", unit=\"V\")\n",
    "\n",
    "display(HTML(\"<h3>Single Load 1 Derived Wave Equations</h3>\"))\n",
    "print(f\"  Event Waveform Equation:   ${eqs_s1['eq_event']}$\")\n",
    "print(f\"  SSD Component Equation:     ${eqs_s1['eq_ssd']}$\")\n",
    "print(f\"  Harmonic Component Eq:      ${eqs_s1['eq_harmonic']}$\")\n",
    "print(f\"  Transient Component Eq:     ${eqs_s1['eq_transient']}$\")\n",
    "\n",
    "display(HTML(\"<h3>Joint Load Pair Derived Wave Equations</h3>\"))\n",
    "print(f\"  Event Waveform Equation:   ${eqs_pair['eq_event']}$\")\n",
    "print(f\"  SSD Component Equation:     ${eqs_pair['eq_ssd']}$\")\n",
    "print(f\"  Harmonic Component Eq:      ${eqs_pair['eq_harmonic']}$\")\n",
    "print(f\"  Transient Component Eq:     ${eqs_pair['eq_transient']}$\")\n",
    "\n",
    "fig_comp, ax_c = plt.subplots(4, 1, figsize=(14, 11), sharex=True)\n",
    "ax_c[0].plot(t, decomp_s1[\"x_event\"], \":\", color=\"navy\", label=\"Single Load 1 Residual x_single1(t)\")\n",
    "ax_c[0].plot(t, decomp_pair[\"x_event\"], \"-\", color=\"black\", linewidth=1.5, label=\"Joint Load Pair Residual x_pair(t)\")\n",
    "ax_c[0].set_ylabel(\"Residual (V)\", fontweight=\"bold\")\n",
    "ax_c[0].set_title(\"Residual Event Signal Comparison: Single Load 1 vs Joint Load Pair\", fontweight=\"bold\")\n",
    "ax_c[0].grid(True, linestyle=\":\", alpha=0.6)\n",
    "ax_c[0].legend(loc=\"upper right\")\n",
    "\n",
    "ax_c[1].plot(t, decomp_s1[\"x_ssd\"], \":\", color=\"navy\", label=\"Single Load 1 x_SSD(t)\")\n",
    "ax_c[1].plot(t, decomp_pair[\"x_ssd\"], \"-\", color=\"darkred\", linewidth=1.8, label=\"Joint Load Pair x_SSD(t)\")\n",
    "ax_c[1].set_ylabel(\"SSD (V)\", fontweight=\"bold\")\n",
    "ax_c[1].grid(True, linestyle=\":\", alpha=0.6)\n",
    "ax_c[1].legend(loc=\"upper right\")\n",
    "\n",
    "ax_c[2].plot(t, decomp_s1[\"x_harmonic\"], \":\", color=\"navy\", label=\"Single Load 1 x_harmonic(t)\")\n",
    "ax_c[2].plot(t, decomp_pair[\"x_harmonic\"], \"-\", color=\"darkblue\", linewidth=1.5, label=\"Joint Load Pair x_harmonic(t)\")\n",
    "ax_c[2].set_ylabel(\"Harmonics (V)\", fontweight=\"bold\")\n",
    "ax_c[2].grid(True, linestyle=\":\", alpha=0.6)\n",
    "ax_c[2].legend(loc=\"upper right\")\n",
    "\n",
    "ax_c[3].plot(t, decomp_s1[\"x_transient\"], \":\", color=\"navy\", label=\"Single Load 1 x_transient(t)\")\n",
    "ax_c[3].plot(t, decomp_pair[\"x_transient\"], \"-\", color=\"darkgreen\", linewidth=1.5, label=\"Joint Load Pair x_transient(t)\")\n",
    "ax_c[3].set_ylabel(\"Transient (V)\", fontweight=\"bold\")\n",
    "ax_c[3].set_xlabel(\"Time (s)\", fontweight=\"bold\")\n",
    "ax_c[3].grid(True, linestyle=\":\", alpha=0.6)\n",
    "ax_c[3].legend(loc=\"upper right\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

stage2_cells = [
    stage2_md,
    cell_1_md,
    cell_1_code,
    cell_2_md,
    cell_2_code,
    cell_3_md,
    cell_3_code
]

# Remaining cells from original notebook (cells 5 to end: equipment group plots, Stage 2 statistical analysis, etc.)
rest_cells = nb["cells"][5:]

nb["cells"] = stage1_cells + stage2_cells + rest_cells

with open("report.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Inserted Stage 2 3-cell clean experiment into report.ipynb cleanly.")
