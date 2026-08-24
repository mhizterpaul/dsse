# 1. Known Upstream Distribution-Station Parameters

The upstream station is the **known boundary condition** of the latent-realization problem. The swing bus is modeled as an ideal infinite bus, so its voltage is fixed and its source strength is infinite.

## 1.1 Swing / utility source

| Parameter | Symbol | Specification |
| --- | --- | --- |
| Bus type | $B_{\mathrm{type}}$ | Swing / slack / infinite bus |
| Nominal voltage | $V_{\mathrm{src}}$ | 33 kV line-to-line RMS |
| Phase sequence | $S_{\mathrm{phase}}$ | ABC |
| Nominal frequency | $f_0$ | 50 Hz |
| Voltage angle reference | $\theta_{\mathrm{src}}$ | $0^\circ$ |
| Source impedance | $Z_{\mathrm{src}}$ | $0 + j0\ \Omega$ ideal infinite bus |
| Source positive-sequence impedance | $Z_1$ | $0\ \Omega$ |
| Source negative-sequence impedance | $Z_2$ | $0\ \Omega$ |
| Source zero-sequence impedance | $Z_0$ | $0\ \Omega$ |

Thus, mathematically,

$$V_{\mathrm{src}}=33\ {\rm kV_{LL}},\qquad \theta_{\mathrm{src}}=0^\circ,\qquad Z_{\mathrm{src}}=0.$$

The source does **not** have an operating-point (P/Q) parameter in the network specification because an infinite bus supplies or absorbs whatever active/reactive power is required to maintain its prescribed voltage.

## 1.2 Distribution substation transformer

For a 33/11-kV upstream distribution station:

| Parameter | Symbol | Specification |
| --- | --- | --- |
| Rated apparent power | $S_{\mathrm{tr,sub}}$ | 7.5 MVA |
| Primary voltage | $V_{HV}$ | 33 kV |
| Secondary voltage | $V_{MV}$ | 11 kV |
| Frequency | $f_0$ | 50 Hz |
| Vector group | $VG$ | Dyn11 |
| Phase count | $n_\phi$ | 3 |
| Winding connection | $Conn$ | HV Delta / LV-MV Wye |
| Neutral grounding | $G_{\mathrm{neut}}$ | MV neutral grounded |
| Percentage impedance | $Z_{\%}$ | 8.35% |
| Winding resistance | $R_{\mathrm{tr}}$ | 0.60% (50 kW copper loss) |
| Leakage reactance | $X_{\mathrm{tr}}$ | 8.33% |
| Core loss | $P_{\mathrm{core}}$ | 0.10% (7.5 kW core loss) |
| Magnetizing reactance | $X_m$ | 250 pu |
| Core-loss resistance | $R_c$ | 800 pu |

## 1.3 Known MV feeder parameters

For every known feeder ($f$):

| Parameter | Symbol | Feeder 1 | Feeder 2 | Feeder 3 |
| --- | --- | --- | --- | --- |
| Feeder ID | $f_i$ | `feeder_1` | `feeder_2` | `feeder_3` |
| Sending bus | $b_s$ | `main_bus` | `main_bus` | `main_bus` |
| Receiving/boundary bus | $b_r$ | `feeder1_head` | `feeder2_head` | `feeder3_head` |
| Nominal voltage | $V_{MV}$ | 11 kV | 11 kV | 11 kV |
| Phase count | $n_\phi$ | 3 | 3 | 3 |
| Conductor type | $C_{\mathrm{type}}$ | Overhead 3-phase | Overhead 3-phase | Overhead 3-phase |
| Conductor resistance | $r_{MV}$ | 0.25 $\Omega$/km | 0.25 $\Omega$/km | 0.25 $\Omega$/km |
| Conductor reactance | $x_{MV}$ | 0.35 $\Omega$/km | 0.35 $\Omega$/km | 0.35 $\Omega$/km |
| Conductor susceptance | $b_{MV}$ | 12.0 $\mu\mathrm{S}$/km | 12.0 $\mu\mathrm{S}$/km | 12.0 $\mu\mathrm{S}$/km |
| Positive-sequence impedance | $Z_{1,MV}$ | $0.25 + j0.35\ \Omega$/km | $0.25 + j0.35\ \Omega$/km | $0.25 + j0.35\ \Omega$/km |
| Zero-sequence impedance | $Z_{0,MV}$ | $0.75 + j1.12\ \Omega$/km | $0.75 + j1.12\ \Omega$/km | $0.75 + j1.12\ \Omega$/km |
| Line length | $L_{MV}$ | 4.5 km | 6.2 km | 8.5 km |
| Number of phases | $n_\phi$ | 3 | 3 | 3 |
| Phase arrangement / spacing | $Arr_{\mathrm{phase}}$ | Horizontal / Standard | Horizontal / Standard | Horizontal / Standard |
| Line configuration | $Cfg_{\mathrm{line}}$ | Overhead | Overhead | Overhead |
