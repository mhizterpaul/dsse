# Time-Adjusted Cluster Load Allocation with Error Correction in Sparsely Metered Distribution Networks

## Methodology

In a sparsely metered distribution system, the utility may know:
- feeder/transformer energy supplied;
- measurements from a relatively small subset of consumers;
- some information about consumer premises;
- historical consumption patterns/classes;
but does not know the actual consumption of every customer.

The formulation is:

$$ E_U = E_F - E_M - E_L $$

where:
- $E_F$: feeder supply energy,
- $E_M$: measured customer energy,
- $E_L$: estimated technical losses (incorporating both transformer losses and line losses),
- $E_U$: energy attributable to unknown/unmetered customers.

Then estimate an expected consumption for each unmetered customer:

$$ w_i = \mathbb{E}[E_i \mid C_i, X_i] $$

where $X_i$ could include:
- customer class,
- historical billing,
- premises characteristics,
- connected load,
- time of year,
- supply availability
- feeder characteristics.

Then:

$$ \boxed{ \hat{E}_i = E_U \frac{w_i}{\sum_{j\in U}w_j} } $$

with time-adjusted consumer/load-class information.

Suppose consumer $i$ belongs to class $c$, with metered class profile $\mu_c(t)$.

For an unmetered consumer, rather than assigning a static class average, estimate:

$$ \hat{E}_i = \int_{t_0}^{t_1} \alpha_i(t)\mu_{c_i}(t)\,dt $$

where $\alpha_i(t)$ is your time adjustment factor for observed metered-class behaviour.

Let the actual feeder energy be:

$$ E_F = E_L + E_{\mathrm{NTL}} + E_T $$

where:
- $E_L$: legitimate consumer consumption;
- $E_{\mathrm{NTL}}$: technical network losses (transformer and line losses);
- $E_T$: non-technical losses/theft.

Therefore the allocation error is:

$$ E_F - \hat{E}_L = \hat{E}_{\mathrm{loss}} + \hat{E}_T $$

We report the baseline CLA error and time-adjusted CLA error, and derive a transient-assisted CLA error correction factor.

### System Model

#### 1. Known Plant for Latent Network Realization

The upstream distribution station is completely known and serves as the boundary for observing downstream states.
It consists of:

```text
        Utility Source (Swing Bus)
                  │
      Distribution Substation Transformer
                  │
        Main Distribution Bus ── Generator
                  │
      ┌───────────┼───────────┐
      │           │           │
    Feeder 1    Feeder 2    Feeder 3
      │           │           │
 Distribution  Distribution  Distribution
 Transformer   Transformer   Transformer
      │           │           │
     LV           LV          LV
 Distribution Distribution Distribution
  Networks     Networks     Networks
```

The plant model contains strictly distribution network elements and local sources to facilitate Latent Network Realization:

* **Utility Source (Swing Bus)**: Represents the steady connection to the transmission grid.
* **Distribution Substation Transformer**: Substation transformer supplying the medium-voltage bus.
* **Main Feeder**: with lines extending from the substation, each characterized by known feeder lengths and impedances.
* **Fixed Set of Transformers**: Step-down distribution transformers whose primary-side terminals serve as the boundary measurement interfaces.
* **Measurement and Monitoring Devices**: Electrical sensors capturing voltage, current, active/reactive power, and sequence components at the meters and transformer primary terminal.
* **Consumer Load Circuits**: To accurately represent realistic residential, commercial, and industrial end-user devices, consumer equipment circuits are implemented compatibly across OpenDSS and ATP-EMTP:
  1. **AC Motor (`ac_motor`)**: Three-phase induction motor with stator resistance/inductance, magnetizing branch, rotor resistance/inductance, and mechanical inertia.
  2. **DC Motor + Inverter (`dc_motor_inverter`)**: Rectifier stage, DC-link capacitor, PWM H-bridge inverter, and DC motor armature $R_a, L_a$ with speed-dependent Back-EMF.
  3. **Microwave (`microwave`)**: Input rectifier, PFC stage, DC-link capacitor, high-voltage transformer, diode voltage doubler, and magnetron non-linear load.
  4. **Induction Plate (`induction_plate`)**: Input rectifier, DC-link, high-frequency resonant inverter, resonant capacitor, and induction coil $R_{\mathrm{eq}} + j\omega L_{\mathrm{eq}}$.
  5. **Compressor (`compressor`)**: Single-phase AC induction motor driving reciprocating/scroll compressor load torque.
  6. **Audio Amplifier (`audio_amplifier`)**: AC supply rectifier, DC-link supply capacitor bank, Class-D switching H-bridge, LC output filter, and speaker impedance.
  7. **Uninterruptible Power Supply / UPS (`ups`)**: Battery bank equivalent circuit, DC-link, bidirectional converter, and AC-side filter interface.
  8. **Industrial Fan (`industrial_fan`)**: Three-phase induction motor driving speed-squared aerodynamic fan load torque.

#### 2. Measurement Architecture

Measurements are obtained from two sensing layers: smart meters measurement at consumer and feeder edge and transformer edge transient analyzer.

1. Smart-Meter Measurements
The metering hierarchy is organized as follows:

```text
               Known MV feeder
                      │
                  Edge Meter
                      │
                ┌─────┴─────┐
                │Transformer│
                └─────┬─────┘
                      │
                      │
             ┌────────┴────────┐
             │                 │
            Smart              |
            Meter              |
             │                 │
           Consumer          Consumer
            Unit A            Unit B
```

Selected candidate units are instrumented with smart meters to acquire:
  Active power (P)
  Reactive power (Q)
  Apparent power (S)
  Power factor (PF)
  Energy consumption (kWh)

2. Transformer Measurements
Each distribution transformer serves as an edge measurement node representing the interface to the downstream network. Measurements include:
Primary Electrical Measurements
  phase voltage magnitude and phase angle
  phase current magnitude and phase angle
  Active power
  Reactive power
  Apparent power
  Power factor

Dynamic Quantities
  Loading rate
  Transformer temperature
  Transient voltage and current waveforms

#### 3. Distribution Network Simulation

The simulation involves first assigning consumer load classes to consumer load circuits. The 3 LV transformer models have fixed varied specifications as detailed in `docs/specs/lv1/lv_transformer.md`, `docs/specs/lv2/lv_transformer.md`, and `docs/specs/lv3/lv_transformer.md`. We take energy consumption from the metered consumer load circuits for time $dt$, and construct Dataset 1 which includes the assigned classes and the energy consumption of the metered group in the network. We compute baseline CLA error and time-adjusted CLA error, considering non-technical losses included in the model. We generated 3 datasets including consumer load circuit switch transient co-events under 3 network conditions, and analyze the observability of these events from which we compute the error correction factor of transformer transients-based consumer load prediction on time-adjusted CLA error. The simulation is performed using OpenDSS and ATP-EMTP.

1. **Dataset 1**: Focuses on Cluster Load Allocation (CLA) energy estimation.
   - **Ground Truth Variables:** `gt_consumer_unit_id`, `consumer_type` (`known` for registered units, `unknown` for latent units), `consumer_unit_source` (bus and line connected to), `consumer_unit_loads`, `assigned_weight` (weight $w_i = \mathbb{E}[E_i \mid C_i, X_i]$ populated for known unmetered units), `gt_consumed_energy_kwh` (populated for known units, empty/NaN for unknown units), `consumer_line_losses`.
   - **Observed Variables:** `measured_energy_kwh` (populated for metered units, empty for unmetered/unknown), `cla_estimates` (populated for unmetered known units, empty for metered/unknown), `time_adjusted_cla_estimates` (populated for unmetered known units, empty for metered/unknown).

2. **Dataset 2**: Evaluates what type of co-events are observable across load switch pairs (`load_load`) and mixed load switch and fault pairs (`load_fault`).
   - **Ground Truth Variables:** `load_source`, `fault_info`, `gt_event_1_equipment_type`, `gt_event_1_fault_type`, `gt_event_1_start_timestamp_s`, `gt_event_2_equipment_type`, `gt_event_2_fault_type`, `gt_event_2_start_timestamp_s`, `gt_time_offset_s`.
   - **Observed Variables:** Three-phase co-event waveforms (`obs_coevent_v`, `obs_coevent_i`), single-event phase-specific waveforms (`obs_single_event_1/2_v_phase_a/b/c`, `obs_single_event_2/2_i_phase_a/b/c`), composed event responses (`obs_composed_event_v_phase_a/b/c`, `obs_composed_event_i_phase_a/b/c`), phase-specific residual waveforms (`obs_residual_v_phase_a/b/c`, `obs_residual_i_phase_a/b/c`), and scalar residual magnitudes (`residual_voltage_magnitude`, `residual_current_magnitude`). Uses a fixed baseline transformer specification and fixed $t_{\mathrm{offset}} = 0.0\,\mathrm{s}$.

3. **Dataset 3**: Evaluates how residual magnitude in pair varies with time shift operation ($t_{\mathrm{offset}} = 0.0\,\mathrm{s}$ vs $t_{\mathrm{offset}} > 0.0\,\mathrm{s}$) across load switch pairs and mixed load-fault pairs.
   - **Ground Truth Variables:** `load_source`, `fault_info`, `gt_event_1_equipment_type`, `gt_event_1_fault_type`, `gt_event_1_start_timestamp_s`, `gt_event_2_equipment_type`, `gt_event_2_fault_type`, `gt_event_2_start_timestamp_s`, `gt_time_offset_s`.
   - **Observed Variables:** Three-phase co-event waveforms (`obs_coevent_v`, `obs_coevent_i`), single-event phase-specific waveforms (`obs_single_event_1/2_v_phase_a/b/c`, `obs_single_event_2/2_i_phase_a/b/c`), composed event responses (`obs_composed_event_v_phase_a/b/c`, `obs_composed_event_i_phase_a/b/c`), phase-specific residual waveforms (`obs_residual_v_phase_a/b/c`, `obs_residual_i_phase_a/b/c`), and scalar residual magnitudes (`residual_voltage_magnitude`, `residual_current_magnitude`). Uses a fixed baseline transformer specification.

4. **Dataset 4**: Evaluates how transformer specification affects the observability of load switch pairs and mixed load-fault pairs. Featuring varying transformer specifications whose physical, sequence, and capability parameters are explicitly specified in `docs/specs/lv1/lv_transformer.md`, `docs/specs/lv2/lv_transformer.md`, and `docs/specs/lv3/lv_transformer.md`.
   - **Ground Truth Variables:** `gt_feeder_id`, `load_source`, `fault_info`, `gt_event_1_equipment_type`, `gt_event_1_fault_type`, `gt_event_1_start_timestamp_s`, `gt_event_2_equipment_type`, `gt_event_2_fault_type`, `gt_event_2_start_timestamp_s`, `gt_time_offset_s`.
   - **Observed Variables:** Three-phase co-event waveforms (`obs_coevent_v`, `obs_coevent_i`), single-event phase-specific waveforms (`obs_single_event_1/2_v_phase_a/b/c`, `obs_single_event_2/2_i_phase_a/b/c`), composed event responses (`obs_composed_event_v_phase_a/b/c`, `obs_composed_event_i_phase_a/b/c`), phase-specific residual waveforms (`obs_residual_v_phase_a/b/c`, `obs_residual_i_phase_a/b/c`), and scalar residual magnitudes (`residual_voltage_magnitude`, `residual_current_magnitude`). Uses fixed $t_{\mathrm{offset}} = 0.0\,\mathrm{s}$.

#### 4. Observability Tests for LV Network Using Transformer Transients

##### Dataset 2 Event Pair Observability Testing

Factorial ANOVA analysis (`src/statistics/q1_event_pair_analysis.py`) evaluates event pair observability across load switch pairs (`load_load`) and mixed load switch and fault pairs (`load_fault`), measuring $F_{\mathrm{voltage}}, p_{\mathrm{voltage}}$ and $F_{\mathrm{current}}, p_{\mathrm{current}}$. For the six phase-specific single-event voltage and current waveform columns ($k=1\dots 6$), the Pearson correlation $r_k^{\mathrm{raw}}$ is discounted by the ambient noise amplitude ratio:

$$ \eta_{\mathrm{noise}} = 10^{-\mathrm{SNR}/20} = 10^{-35/20} \approx 0.0178 $$

at an assumed ambient noise level of $\mathrm{SNR} = 35\,\mathrm{dB}$, yielding the discounted correlation $r_k = \max(0, r_k^{\mathrm{raw}} - \eta_{\mathrm{noise}})$. The per-event aggregate similarity $S = \frac{1}{6}\sum_{k=1}^6 r_k$, standard deviation $\sigma_r = \sqrt{\frac{1}{5}\sum_{k=1}^6 (r_k - S)^2}$, dataset mean correlation $\bar{r}_2$, and mean dissimilarity $D_2 = 1 - \bar{r}_2$ are then evaluated.

##### Dataset 3 Time Shift Operation Variation Testing

Levene / Brown-Forsythe variance analysis (`src/statistics/q2_time_shift_analysis.py`) evaluates residual magnitude variation under time shift operations ($t_{\mathrm{offset}} = 0.0\,\mathrm{s}$ vs $t_{\mathrm{offset}} > 0.0\,\mathrm{s}$) using Dataset 3 across load switch pairs and mixed load-fault pairs. The waveform Pearson correlation statistics ($\bar{r}_3$, $\sigma_{r,3}$, and dissimilarity $D_3 = 1 - \bar{r}_3$) incorporating the $35\,\mathrm{dB}$ SNR noise discount quantify signal consistency under temporal offsets.

##### Dataset 4 Transformer Specification Effect Testing

One-Way ANOVA testing (`src/statistics/q3_transformer_spec_analysis.py`) evaluates how transformer specification variations affect observability across load switch pairs and mixed pairs, measuring $F_{\mathrm{spec}}, p_{\mathrm{spec}}$ across transformer specifications alongside waveform Pearson correlation statistics ($\bar{r}_4$, $\sigma_{r,4}$, and dissimilarity $D_4 = 1 - \bar{r}_4$) with the $35\,\mathrm{dB}$ SNR noise discount.

##### Transient-Assisted Error Reduction Factor

The overall mean dissimilarity $D = 1 - \bar{r}_{\mathrm{mean}}$ aggregated across Datasets 2, 3, and 4 (which naturally embeds the conservative $35\,\mathrm{dB}$ SNR uncertainty allowance $\eta_{\mathrm{noise}}$ per waveform column) directly provides the transient error reduction factor, which is applied to scale and correct the time-adjusted Cluster Load Allocation (CLA) estimation error.

**Limitations:** The validation establishes the practical limits of boundary-based realization and identifies the sensing architecture required for distributed dynamic state estimation in partially observable distribution networks within the limits of the simulated environment.
