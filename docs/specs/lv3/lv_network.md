# 2. LV Network Parameters - LV Feeder 3

## 2.1 LV network-level parameters

| Parameter | Symbol | Specification |
| --- | --- | --- |
| Network ID | $N_i$ | `LV3` |
| Topology type | $T_{\mathrm{type}}$ | Radial |
| Number of buses | $N_b$ | 30 |
| Number of branches | $N_l$ | 29 |
| Number of phases | $n_\phi$ | 3 |
| Nominal LV voltage | $V_{LV}$ | $415\ \mathrm{V_{LL}}$ / $240\ \mathrm{V_{LN}}$ |
| Phase configuration | $Cfg_{\mathrm{phase}}$ | 3-phase |
| Neutral configuration | $Cfg_{\mathrm{neut}}$ | 4-wire |
| Grounding configuration | $Cfg_{\mathrm{grnd}}$ | Solidly Grounded |
| Base frequency | $f_0$ | 50 Hz |
| Network base power | $S_{\mathrm{base}}$ | 2.0 MVA |
| Network base voltage | $V_{\mathrm{base}}$ | 415 V |

## 2.2 LV topology parameters

For each bus in LV Network 3:

| Bus ID ($b_i$) | Phase Availability ($A_i$) | Bus Type ($T_{\mathrm{bus}}$) | Parent Bus ($b_p$) | Grounding Connection ($G_{\mathrm{conn}}$) |
| --- | --- | --- | --- | --- |
| `feeder3_sec` | $\{A,B,C,N\}$ | Root / LV Secondary | `feeder3_head` | Solidly Grounded |
| `f3_node1` | $\{A,B,C,N\}$ | Intermediate | `feeder3_sec` | Grounded |
| `f3_node2` | $\{A,B,C,N\}$ | Intermediate | `feeder3_sec` | Grounded |
| `f3_node3` ... `f3_node29` | $\{A,B,C,N\}$ | Consumer Load Node | Known Tree Parent | Grounded |
