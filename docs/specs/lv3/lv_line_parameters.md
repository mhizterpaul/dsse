# 3. LV Line Parameters - LV Network 3

For every LV line in Network 3:

## Mandatory parameters

| Parameter | Symbol | Unit | Specification |
| --- | --- | --- | --- |
| Line length | $L$ | km | 0.05 – 0.08 km |
| Conductor type | $C_{\mathrm{type}}$ | Unitless | 150 mm² All-Aluminum Conductor (AAC) |
| Thermal rating / Ampacity | $I_{\mathrm{norm}}$ | A | 350 A per phase |
| Number of phases | $n_\phi$ | Unitless | 3 |
| Conductor resistance | $r$ | $\Omega/\mathrm{km}$ | 0.21 $\Omega$/km |
| Conductor reactance | $x$ | $\Omega/\mathrm{km}$ | 0.08 $\Omega$/km |
| Conductance | $g$ | S/km | $1.0 \times 10^{-6}$ S/km |
| Susceptance | $b$ | S/km | $1.0 \times 10^{-6}$ S/km |
| Positive-sequence resistance ($R_1$) | $R_1$ | $\Omega/\mathrm{km}$ | 0.21 $\Omega$/km |
| Positive-sequence reactance ($X_1$) | $X_1$ | $\Omega/\mathrm{km}$ | 0.08 $\Omega$/km |
| Zero-sequence resistance ($R_0$) | $R_0$ | $\Omega/\mathrm{km}$ | 0.63 $\Omega$/km |
| Zero-sequence reactance ($X_0$) | $X_0$ | $\Omega/\mathrm{km}$ | 0.24 $\Omega$/km |

## Known Branch Segment Inventory

For lines `down_3_1` through `down_3_29`:

| Branch ID ($l_i$) | From bus ($b_f$) | To bus ($b_t$) | Length ($L_l$) | Series $R_l = r L_l$ | Series $X_l = x L_l$ | Shunt $G_l = g L_l$ | Shunt $B_l = b L_l$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `down_3_1` | `feeder3_sec` | `f3_node1` | 0.05 km | 0.0105 $\Omega$ | 0.0040 $\Omega$ | $5.0 \times 10^{-8}$ S | $5.0 \times 10^{-8}$ S |
| `down_3_2` | `feeder3_sec` | `f3_node2` | 0.06 km | 0.0126 $\Omega$ | 0.0048 $\Omega$ | $6.0 \times 10^{-8}$ S | $6.0 \times 10^{-8}$ S |
| `down_3_3` ... `down_3_29` | Known Parent | Known Child | 0.05–0.08 km | $0.21 L_l\ \Omega$ | $0.08 L_l\ \Omega$ | $10^{-6} L_l\ \mathrm{S}$ | $10^{-6} L_l\ \mathrm{S}$ |
