import numpy as np
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from src.power_plant.hv_transformer import build_hv_transformer
from src.power_plant.lv_transformers import get_distribution_transformer_spec
from src.power_plant.lv_network_1 import generate_lv1_topology, build_lv1_network, register_lv1_consumers
from src.power_plant.lv_network_2 import generate_lv2_topology, build_lv2_network, register_lv2_consumers
from src.power_plant.lv_network_3 import generate_lv3_topology, build_lv3_network, register_lv3_consumers
from src.power_plant.consumer_registry import ConsumerRegistry
from src.loads import get_equipment_model


def generate_known_radial_topology(feeder_idx: int, num_buses: int = 20, rng=None, seed: int = 42) -> dict:
    """
    Generates deterministic known radial tree topologies (LV1, LV2, LV3) aligned with docs/specs.
    """
    if feeder_idx == 1:
        return generate_lv1_topology(seed=1000 + seed + feeder_idx)
    elif feeder_idx == 2:
        return generate_lv2_topology(seed=1000 + seed + feeder_idx)
    elif feeder_idx == 3:
        return generate_lv3_topology(seed=1000 + seed + feeder_idx)
    else:
        return generate_lv1_topology(seed=1000 + seed + feeder_idx)


generate_radial_topology = generate_known_radial_topology


def identify_candidate_consumer_units(topology: dict) -> list[dict]:
    """
    Identifies candidate consumer units and edge transformer consumer units across the known LV network.
    """
    candidate_consumer_units = []

    topologies = topology.get("topologies", {})
    if topologies:
        for f_id, sub_topo in topologies.items():
            for ln in sub_topo.get("lines", []):
                parent = ln["bus1"]
                child = ln["bus2"]
                line_name = ln["name"]

                candidate_consumer_units.append({
                    "consumer_unit_id": f"consumer_unit_{line_name}",
                    "bus": child,
                    "parent_bus": parent,
                    "branch_id": line_name,
                    "branch_type": "consumer_line",
                    "consumer_eligible": True
                })
    else:
        for ln in topology.get("lines", []):
            parent = ln["bus1"]
            child = ln["bus2"]
            line_name = ln["name"]

            candidate_consumer_units.append({
                "consumer_unit_id": f"consumer_unit_{line_name}",
                "bus": child,
                "parent_bus": parent,
                "branch_id": line_name,
                "branch_type": "consumer_line",
                "consumer_eligible": True
            })

    # LV secondary terminals of the distribution transformers (Feeder boundary consumer_units)
    for idx in [1, 2, 3]:
        candidate_consumer_units.append({
            "consumer_unit_id": f"trans{idx}_lv_boundary_consumer_unit",
            "bus": f"feeder{idx}_sec",
            "parent_bus": f"feeder{idx}_head",
            "branch_id": f"transformer.trans{idx}",
            "branch_type": "transformer_boundary",
            "consumer_eligible": True
        })

    return candidate_consumer_units


@dataclass
class OperatingPoint:
    time_s: float
    generator_p_kw: float
    generator_q_kvar: float
    feeder_p_kw: dict
    feeder_q_kvar: dict
    transformer_loading: dict
    voltage_pu: dict
    frequency_hz: float
    transient_waveforms: Optional[object] = None
    phase_voltages_v: Optional[dict] = None
    phase_angles_deg: Optional[dict] = None

    def import_atp_cases(self, atp_waveforms):
        self.transient_waveforms = atp_waveforms


def configure_generator(dss, p_kw: float = 1500.0, q_kvar: float = 0.0):
    """
    Configures the generator element in OpenDSS.
    """
    dss.run_command(f"new generator.gen1 bus1=sourcebus phases=3 kv=33.0 kw={p_kw} kvar={q_kvar} model=1")


def apply_generator_profile(dss, p_kw: Optional[float] = None, q_kvar: Optional[float] = None):
    """
    Applies or updates generator active and reactive power profiles in OpenDSS if specified.
    """
    if dss.Circuit.SetActiveElement("Generator.gen1"):
        if p_kw is not None:
            dss.run_command(f"Generator.gen1.kw={p_kw}")
        if q_kvar is not None:
            dss.run_command(f"Generator.gen1.kvar={q_kvar}")


def compute_generator_size(
    registry: Optional[ConsumerRegistry] = None,
    safety_factor: float = 1.25,
    power_factor: float = 0.8
) -> dict:
    """
    Computes generator size suitable for the network during initialization in plant.py
    using the loads in the network and transformers as described in the 4-step algorithm:

    Step 1: List appliances, find running watts (steady power) and starting watts
            (extra power needed for motor-driven items). Add up running watts.
    Step 2: Account for power surges by finding the single appliance with highest starting watts surge
            (starting watts - running watts) and add extra surge amount to total running watts.
    Step 3: Multiply total wattage by safety buffer (1.20 to 1.25) to prevent overloading.
            Divide final wattage by 1000 to get kilowatts (kW).
    Step 4: Divide total kW by standard power factor of 0.8 to find minimum kVA.
    """
    total_running_watts = 0.0
    max_starting_surge_watts = 0.0

    motor_surge_multipliers = {
        "ac_motor": 3.0,
        "compressor": 3.5,
        "industrial_fan": 2.5,
        "dc_motor_inverter": 2.0,
    }

    if registry is not None:
        for unit in registry.get_all_consumers():
            for ld in unit.loads:
                try:
                    eq_model = get_equipment_model(ld.load_type)
                    running_w = float(eq_model.rated_power_kw) * 1000.0
                except Exception:
                    running_w = 10000.0

                total_running_watts += running_w

                mult = motor_surge_multipliers.get(ld.load_type, 1.0)
                starting_w = running_w * mult
                surge_w = starting_w - running_w
                if surge_w > max_starting_surge_watts:
                    max_starting_surge_watts = surge_w

    # Step 2: Account for Power Surges
    peak_watts = total_running_watts + max_starting_surge_watts

    # Step 3: Add Safety Margin (20%-25%) and convert to kW
    design_watts = peak_watts * safety_factor
    generator_kw = design_watts / 1000.0

    # Step 4: Convert to kVA using standard power factor 0.8
    generator_kva = generator_kw / power_factor

    return {
        "total_running_watts": total_running_watts,
        "max_starting_surge_watts": max_starting_surge_watts,
        "peak_watts": peak_watts,
        "generator_kw": round(generator_kw, 2),
        "generator_kva": round(generator_kva, 2)
    }


def build_single_lv_network_composition(
    dss,
    feeder_idx: int = 1,
    generator_p_kw: Optional[float] = None,
    generator_q_kvar: float = 0.0,
    use_baseline_transformers: bool = True,
    loads_dict: dict = None,
    seed: int = 42,
    verbose: bool = False
) -> dict:
    """
    Composes Case 1: Single LV network configuration (1 LV feeder network).
    """
    top = generate_known_radial_topology(feeder_idx, seed=seed)
    single_topology = {"topologies": {feeder_idx: top}}

    dss.run_command("clear")
    dss.run_command("new circuit.plant_substation basekv=33.0 pu=1.0 phases=3 bus1=sourcebus basefreq=50.0")

    build_hv_transformer(dss, verbose=verbose)

    dss.run_command(
        "new linecode.mv_feeder_linecode nphases=3 r1=0.25 x1=0.35 r0=0.75 x0=1.12 c1=0.03819 c0=0.012 units=km normamps=400.0"
    )

    mv_feeder_lengths = {1: 4.5, 2: 6.2, 3: 8.5}
    for f_id, length in mv_feeder_lengths.items():
        dss.run_command(
            f"new line.mv_feeder_{f_id} bus1=main_bus bus2=feeder{f_id}_head phases=3 "
            f"linecode=mv_feeder_linecode length={length} units=km"
        )

    for f_id in [1, 2, 3]:
        spec = get_distribution_transformer_spec(f_id, use_baseline=use_baseline_transformers)
        dss.run_command(
            f"new transformer.{spec['name']} "
            f"phases={spec['phases']} windings={spec['windings']} "
            f"buses=[{','.join(spec['buses'])}] "
            f"conns=[{','.join(spec['conns'])}] "
            f"kvs=[{','.join(map(str, spec['kvs']))}] "
            f"kvas=[{','.join(map(str, spec['kvas']))}] "
            f"%r={spec['r_pct']} "
            f"xhl={spec['xhl_pct']} "
            f"%noloadloss={spec.get('noloadloss_pct', 0.1)} "
            f"%imag={spec.get('imag_pct', 0.8)}"
        )

    registry = ConsumerRegistry(seed=seed)
    if feeder_idx == 1:
        register_lv1_consumers(topology=top, seed=seed, registry=registry)
        build_lv1_network(dss, topology=top, loads_dict=loads_dict, registry=registry, seed=seed)
    elif feeder_idx == 2:
        register_lv2_consumers(topology=top, seed=seed, registry=registry)
        build_lv2_network(dss, topology=top, loads_dict=loads_dict, registry=registry, seed=seed)
    else:
        register_lv3_consumers(topology=top, seed=seed, registry=registry)
        build_lv3_network(dss, topology=top, loads_dict=loads_dict, registry=registry, seed=seed)

    generator_info = compute_generator_size(registry)
    if generator_p_kw is None or generator_p_kw <= 0:
        generator_p_kw = generator_info["generator_kw"]

    if generator_p_kw > 0:
        configure_generator(dss, p_kw=generator_p_kw, q_kvar=generator_q_kvar)

    dss.run_command("solve")
    candidate_consumer_units = identify_candidate_consumer_units(single_topology)

    return {
        "topology": single_topology,
        "registry": registry,
        "candidate_consumer_units": candidate_consumer_units,
        "generator_info": generator_info
    }


def build_three_lv_networks_composition(
    dss,
    generator_p_kw: Optional[float] = None,
    generator_q_kvar: float = 0.0,
    use_baseline_transformers: bool = True,
    loads_dict: dict = None,
    seed: int = 42,
    verbose: bool = False
) -> dict:
    """
    Composes Case 2: Three LV networks configuration (LV1, LV2, LV3 networks).
    """
    top1 = generate_lv1_topology(seed=seed + 1)
    top2 = generate_lv2_topology(seed=seed + 2)
    top3 = generate_lv3_topology(seed=seed + 3)

    combined_topology = {
        "topologies": {
            1: top1,
            2: top2,
            3: top3
        }
    }

    dss.run_command("clear")
    dss.run_command("new circuit.plant_substation basekv=33.0 pu=1.0 phases=3 bus1=sourcebus basefreq=50.0")

    build_hv_transformer(dss, verbose=verbose)

    dss.run_command(
        "new linecode.mv_feeder_linecode nphases=3 r1=0.25 x1=0.35 r0=0.75 x0=1.12 c1=0.03819 c0=0.012 units=km normamps=400.0"
    )

    mv_feeder_lengths = {1: 4.5, 2: 6.2, 3: 8.5}
    for f_id, length in mv_feeder_lengths.items():
        dss.run_command(
            f"new line.mv_feeder_{f_id} bus1=main_bus bus2=feeder{f_id}_head phases=3 "
            f"linecode=mv_feeder_linecode length={length} units=km"
        )

    for f_id in [1, 2, 3]:
        spec = get_distribution_transformer_spec(f_id, use_baseline=use_baseline_transformers)
        dss.run_command(
            f"new transformer.{spec['name']} "
            f"phases={spec['phases']} windings={spec['windings']} "
            f"buses=[{','.join(spec['buses'])}] "
            f"conns=[{','.join(spec['conns'])}] "
            f"kvs=[{','.join(map(str, spec['kvs']))}] "
            f"kvas=[{','.join(map(str, spec['kvas']))}] "
            f"%r={spec['r_pct']} "
            f"xhl={spec['xhl_pct']} "
            f"%noloadloss={spec.get('noloadloss_pct', 0.1)} "
            f"%imag={spec.get('imag_pct', 0.8)}"
        )

    registry = ConsumerRegistry(seed=seed)
    register_lv1_consumers(topology=top1, seed=seed + 1, registry=registry)
    register_lv2_consumers(topology=top2, seed=seed + 2, registry=registry)
    register_lv3_consumers(topology=top3, seed=seed + 3, registry=registry)

    build_lv1_network(dss, topology=top1, loads_dict=loads_dict, registry=registry, seed=seed + 1)
    build_lv2_network(dss, topology=top2, loads_dict=loads_dict, registry=registry, seed=seed + 2)
    build_lv3_network(dss, topology=top3, loads_dict=loads_dict, registry=registry, seed=seed + 3)

    generator_info = compute_generator_size(registry)
    if generator_p_kw is None or generator_p_kw <= 0:
        generator_p_kw = generator_info["generator_kw"]

    if generator_p_kw > 0:
        configure_generator(dss, p_kw=generator_p_kw, q_kvar=generator_q_kvar)

    dss.run_command("solve")
    candidate_consumer_units = identify_candidate_consumer_units(combined_topology)

    return {
        "topology": combined_topology,
        "registry": registry,
        "candidate_consumer_units": candidate_consumer_units,
        "generator_info": generator_info
    }


build_power_plant_and_downstream_networks = build_three_lv_networks_composition


def solve_operating_point(
    dss,
    p_kw: Optional[float] = None,
    q_kvar: Optional[float] = None,
    time_s: float = 0.0
) -> OperatingPoint:
    """
    Applies generator profiles, runs OpenDSS power flow, and extracts electrical operating point.
    Evaluates system operating frequency dynamically from OpenDSS.
    """
    if p_kw is not None or q_kvar is not None:
        apply_generator_profile(dss, p_kw=p_kw, q_kvar=q_kvar)

    dss.run_command("solve")

    # Evaluate operating frequency dynamically from OpenDSS solution
    try:
        freq_hz = float(dss.Solution.Frequency())
        if freq_hz <= 0:
            print(f"ERROR: OpenDSS returned invalid non-positive frequency ({freq_hz} Hz)\n{traceback.format_exc()}")
            raise ValueError(f"OpenDSS solution returned invalid frequency: {freq_hz} Hz")
    except Exception as e:
        print(f"ERROR: Could not retrieve frequency from OpenDSS Solution ({e})\n{traceback.format_exc()}")
        raise RuntimeError(f"Failed to evaluate system operating frequency from OpenDSS: {e}") from e

    gen_p = p_kw if p_kw is not None else 0.0
    gen_q = q_kvar if q_kvar is not None else 0.0
    if dss.Circuit.SetActiveElement("Generator.gen1"):
        powers = dss.CktElement.Powers()
        if len(powers) >= 2:
            gen_p = round(float(abs(powers[0])), 4)
            gen_q = round(float(abs(powers[1])), 4)
    elif dss.Circuit.SetActiveElement("Vsource.source"):
        powers = dss.CktElement.Powers()
        if len(powers) >= 2:
            gen_p = round(float(abs(powers[0])), 4)
            gen_q = round(float(abs(powers[1])), 4)

    feeder_p = {}
    feeder_q = {}
    loading = {}
    voltage_pu = {}
    phase_voltages_v = {}
    phase_angles_deg = {}

    for idx in [1, 2, 3]:
        tx_name = f"Transformer.trans{idx}"
        if dss.Circuit.SetActiveElement(tx_name):
            powers = dss.CktElement.Powers()
            if len(powers) >= 2:
                feeder_p[idx] = round(float(abs(powers[0])), 4)
                feeder_q[idx] = round(float(abs(powers[1])), 4)
            loading[idx] = round(float(np.sqrt(feeder_p.get(idx, 0.0)**2 + feeder_q.get(idx, 0.0)**2) / 1500.0), 4)

        # Retrieve LV terminal phase voltages and angles
        sec_bus = f"feeder{idx}_sec"
        if dss.Circuit.SetActiveBus(sec_bus):
            v_mag_angle = dss.Bus.VMagAngle()
            if len(v_mag_angle) >= 6:
                phase_voltages_v[f"trans{idx}"] = (float(v_mag_angle[0]), float(v_mag_angle[2]), float(v_mag_angle[4]))
                phase_angles_deg[f"trans{idx}"] = (float(v_mag_angle[1]), float(v_mag_angle[3]), float(v_mag_angle[5]))
                phase_voltages_v[f"feeder{idx}_head"] = (float(v_mag_angle[0]), float(v_mag_angle[2]), float(v_mag_angle[4]))
                phase_angles_deg[f"feeder{idx}_head"] = (float(v_mag_angle[1]), float(v_mag_angle[3]), float(v_mag_angle[5]))

    return OperatingPoint(
        time_s=time_s,
        generator_p_kw=gen_p,
        generator_q_kvar=gen_q,
        feeder_p_kw=feeder_p,
        feeder_q_kvar=feeder_q,
        transformer_loading=loading,
        voltage_pu=voltage_pu,
        frequency_hz=freq_hz,
        phase_voltages_v=phase_voltages_v,
        phase_angles_deg=phase_angles_deg
    )
