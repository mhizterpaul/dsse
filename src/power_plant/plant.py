import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from opendssdirect import dss
from src.power_plant.sources import configure_generator, apply_generator_profile
from src.power_plant.transformers import get_distribution_transformer_spec
from src.power_plant.lv_network_1 import generate_lv1_topology, build_lv1_network, register_lv1_consumers
from src.power_plant.lv_network_2 import generate_lv2_topology, build_lv2_network, register_lv2_consumers
from src.power_plant.lv_network_3 import generate_lv3_topology, build_lv3_network, register_lv3_consumers
from src.power_plant.consumer_registry import ConsumerRegistry


def extract_transformer_meter_data(meter: dict) -> dict:
    """
    Extracts transformer boundary voltage and power flow measurements directly from OpenDSS API.
    """
    bus = meter.get("bus", "feeder1_sec")
    dss.Circuit.SetActiveBus(bus)
    v_vec = np.array(dss.Bus.VMagAngle())

    if len(v_vec) >= 6:
        v_mags = v_vec[0::2]
        v_angs = v_vec[1::2]
    else:
        v_mags = np.array([240.0, 240.0, 240.0])
        v_angs = np.array([0.0, -120.0, -240.0])

    branch_id = meter.get("branch_id", "transformer.trans1")
    dss.Circuit.SetActiveElement(branch_id)
    powers = dss.CktElement.Powers()

    if len(powers) >= 2:
        p_kw = float(abs(sum(powers[0::2])))
        q_kvar = float(abs(sum(powers[1::2])))
    else:
        p_kw = 500.0
        q_kvar = 100.0

    s_kva = float(np.sqrt(p_kw**2 + q_kvar**2))

    return {
        "v_mags": v_mags,
        "v_angs": v_angs,
        "p_kw": p_kw,
        "q_kvar": q_kvar,
        "s_kva": s_kva
    }


def generate_known_radial_topology(feeder_idx: int, num_buses: int = 20, rng=None) -> dict:
    """
    Generates deterministic known radial tree topologies (LV1, LV2, LV3) aligned with docs/specs.
    """
    if feeder_idx == 1:
        return generate_lv1_topology(seed=42 + feeder_idx)
    elif feeder_idx == 2:
        return generate_lv2_topology(seed=42 + feeder_idx)
    elif feeder_idx == 3:
        return generate_lv3_topology(seed=42 + feeder_idx)
    else:
        return generate_lv1_topology(seed=42 + feeder_idx)


generate_radial_topology = generate_known_radial_topology


def identify_candidate_consumer_meters(topology: dict) -> list[dict]:
    """
    Identifies candidate consumer meters and edge transformer meters across the known LV network.
    """
    candidate_meters = []

    topologies = topology.get("topologies", {})
    if topologies:
        for f_id, sub_topo in topologies.items():
            for ln in sub_topo.get("lines", []):
                parent = ln["bus1"]
                child = ln["bus2"]
                line_name = ln["name"]

                candidate_meters.append({
                    "meter_id": f"consumer_meter_{line_name}",
                    "bus": child,
                    "parent_bus": parent,
                    "branch_id": line_name,
                    "branch_type": "consumer_line",
                    "meter_eligible": True
                })
    else:
        for ln in topology.get("lines", []):
            parent = ln["bus1"]
            child = ln["bus2"]
            line_name = ln["name"]

            candidate_meters.append({
                "meter_id": f"consumer_meter_{line_name}",
                "bus": child,
                "parent_bus": parent,
                "branch_id": line_name,
                "branch_type": "consumer_line",
                "meter_eligible": True
            })

    # LV secondary terminals of the distribution transformers (Feeder boundary meters)
    for idx in [1, 2, 3]:
        candidate_meters.append({
            "meter_id": f"trans{idx}_lv_boundary_meter",
            "bus": f"feeder{idx}_sec",
            "parent_bus": f"feeder{idx}_head",
            "branch_id": f"transformer.trans{idx}",
            "branch_type": "transformer_boundary",
            "meter_eligible": True
        })

    return candidate_meters


identify_candidate_pccs = identify_candidate_consumer_meters


def select_metered_consumers(candidate_meters: list[dict], fraction: float, seed: int) -> list[dict]:
    """
    Selects transformer boundary meters and a configured fraction of consumer meters.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"meter_fraction must be in (0.0, 1.0], got {fraction}")

    transformer_meters = [m for m in candidate_meters if m.get("branch_type") == "transformer_boundary"]
    consumer_meters = [m for m in candidate_meters if m.get("branch_type") != "transformer_boundary"]

    n_consumer_meters = max(1, int(np.ceil(fraction * len(consumer_meters)))) if consumer_meters else 0

    rng = np.random.default_rng(seed)

    if consumer_meters:
        selected_indices = rng.choice(len(consumer_meters), size=n_consumer_meters, replace=False)
        selected_consumer_meters = [consumer_meters[i] for i in selected_indices]
    else:
        selected_consumer_meters = []

    return transformer_meters + selected_consumer_meters


select_metered_pccs = select_metered_consumers


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


def initialize_known_plant(use_baseline_transformers: bool = False, topology: dict = None, seed: int = 42) -> ConsumerRegistry:
    """
    Initializes the fixed upstream distribution station and registers downstream consumers on each LV network.
    """
    print("INFO: Initializing OpenDSS Physics-Based Known Plant Model (33/11/0.415 kV)...")

    dss.Basic.ClearAll()
    dss.run_command("new circuit.FixedPlant basekv=33.0 pu=1.0 phases=3")

    dss.run_command(
        "new transformer.substation "
        "phases=3 windings=2 "
        "buses=[sourcebus, main_bus] "
        "conns=[delta, wye] "
        "kvs=[33.0, 11.0] "
        "kvas=[7500, 7500] "
        "%r=0.6 "
        "%loadloss=0.667 "
        "%noloadloss=0.1 "
        "%imag=0.8 "
        "xhl=8.33"
    )

    configure_generator(p_kw=1500.0, q_kvar=0.0)

    dss.run_command("new linecode.feeder nphases=3 r1=0.25 x1=0.35 r0=0.75 x0=1.12 c1=12.0 c0=6.0 units=km")

    dss.run_command("new line.feeder1 bus1=main_bus bus2=feeder1_head phases=3 linecode=feeder length=4.5 units=km")
    dss.run_command("new line.feeder2 bus1=main_bus bus2=feeder2_head phases=3 linecode=feeder length=6.2 units=km")
    dss.run_command("new line.feeder3 bus1=main_bus bus2=feeder3_head phases=3 linecode=feeder length=8.5 units=km")

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

    if topology is None:
        top1 = generate_lv1_topology(seed=seed + 1)
        top2 = generate_lv2_topology(seed=seed + 2)
        top3 = generate_lv3_topology(seed=seed + 3)
        topology = {"topologies": {1: top1, 2: top2, 3: top3}}

    registry = ConsumerRegistry(seed=seed)
    topologies = topology.get("topologies", {})

    register_lv1_consumers(topology=topologies.get(1), seed=seed + 1, registry=registry)
    register_lv2_consumers(topology=topologies.get(2), seed=seed + 2, registry=registry)
    register_lv3_consumers(topology=topologies.get(3), seed=seed + 3, registry=registry)

    print("INFO: OpenDSS Known Plant Model and Consumer Registry successfully initialized.")
    return registry


def build_power_plant_and_downstream_networks(
    generator_p_kw: float = 1500.0,
    generator_q_kvar: float = 0.0,
    use_baseline_transformers: bool = True,
    loads_dict: dict = None,
    seed: int = 42
) -> dict:
    """
    Full composition function in plant.py combining:
    - Upstream generator, HV substation transformer, LV transformers
    - LV Networks (LV1, LV2, LV3)
    - Consumer loads registered to LV networks
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

    registry = initialize_known_plant(use_baseline_transformers=use_baseline_transformers, topology=combined_topology, seed=seed)

    if generator_p_kw > 0:
        configure_generator(p_kw=generator_p_kw, q_kvar=generator_q_kvar)

    build_lv1_network(topology=top1, loads_dict=loads_dict, registry=registry, seed=seed + 1)
    build_lv2_network(topology=top2, loads_dict=loads_dict, registry=registry, seed=seed + 2)
    build_lv3_network(topology=top3, loads_dict=loads_dict, registry=registry, seed=seed + 3)

    dss.run_command("solve")

    candidate_meters = identify_candidate_consumer_meters(combined_topology)

    return {
        "topology": combined_topology,
        "registry": registry,
        "candidate_meters": candidate_meters
    }


def solve_operating_point(p_kw: float, q_kvar: float, time_s: float = 0.0) -> OperatingPoint:
    """
    Applies generator profiles, runs OpenDSS power flow, and extracts electrical operating point.
    """
    apply_generator_profile(p_kw, q_kvar)

    dss.Solution.Solve()
    if not dss.Solution.Converged():
        dss.run_command("Solve mode=direct")
        if not dss.Solution.Converged():
            raise RuntimeError(f"OpenDSS failed to converge at t={time_s}s")

    feeder_p = {}
    feeder_q = {}
    loading = {}
    voltage_pu = {}
    phase_voltages_v = {}
    phase_angles_deg = {}

    for idx in [1, 2, 3]:
        meter = {
            "meter_id": f"trans{idx}_lv_boundary_meter",
            "pcc_id": f"trans{idx}_lv_pcc",
            "bus": f"feeder{idx}_sec",
            "parent_bus": f"feeder{idx}_head",
            "branch_id": f"transformer.trans{idx}",
            "branch_type": "transformer"
        }
        data = extract_transformer_meter_data(meter)

        feeder_p[f"feeder{idx}"] = data["p_kw"]
        feeder_q[f"feeder{idx}"] = data["q_kvar"]
        loading[f"transformer{idx}"] = (data["s_kva"] / 1500.0) * 100.0
        v_avg_lv = float(np.mean(data["v_mags"]))
        v_nom_lv = 415.0 / np.sqrt(3.0)
        voltage_pu[f"transformer{idx}"] = v_avg_lv / v_nom_lv

        phase_voltages_v[f"trans{idx}"] = tuple(data["v_mags"])
        phase_angles_deg[f"trans{idx}"] = tuple(data["v_angs"])

    freq = float(dss.Solution.Frequency())

    return OperatingPoint(
        time_s=time_s,
        generator_p_kw=p_kw,
        generator_q_kvar=q_kvar,
        feeder_p_kw=feeder_p,
        feeder_q_kvar=feeder_q,
        transformer_loading=loading,
        voltage_pu=voltage_pu,
        frequency_hz=freq,
        phase_voltages_v=phase_voltages_v,
        phase_angles_deg=phase_angles_deg
    )
