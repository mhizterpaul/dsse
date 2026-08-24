import numpy as np
from opendssdirect import dss

def compute_symmetrical_components(mags: list[float], angles_deg: list[float]) -> dict:
    """
    Computes zero-, positive-, and negative-sequence phasor quantities from 3-phase magnitudes and angles.
    """
    if len(mags) < 3 or len(angles_deg) < 3:
        return {
            "zero": (0.0, 0.0),
            "positive": (0.0, 0.0),
            "negative": (0.0, 0.0)
        }
    rad = np.radians(angles_deg)
    phasors = [m * (np.cos(r) + 1j * np.sin(r)) for m, r in zip(mags, rad)]

    a = np.cos(2.0*np.pi/3.0) + 1j * np.sin(2.0*np.pi/3.0)
    a_sq = a * a

    x0 = (phasors[0] + phasors[1] + phasors[2]) / 3.0
    x1 = (phasors[0] + a * phasors[1] + a_sq * phasors[2]) / 3.0
    x2 = (phasors[0] + a_sq * phasors[1] + a * phasors[2]) / 3.0

    return {
        "zero": (float(np.abs(x0)), float(np.degrees(np.angle(x0)))),
        "positive": (float(np.abs(x1)), float(np.degrees(np.angle(x1)))),
        "negative": (float(np.abs(x2)), float(np.degrees(np.angle(x2))))
    }

def extract_bus_voltages(bus_name: str):
    """
    Extracts magnitude and phase angles from Bus.VMagAngle() using correct stride slicing.
    """
    dss.Circuit.SetActiveBus(bus_name)
    v_mag_angle = dss.Bus.VMagAngle()

    if not v_mag_angle or len(v_mag_angle) < 6:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    mags = v_mag_angle[0:6:2]
    angles = v_mag_angle[1:6:2]
    return list(mags), list(angles)

def extract_element_currents(element_name: str, terminal: int = 1):
    """
    Extracts currents for terminal 1 or 2 of a specific element.
    Uses correct stride slicing.
    """
    dss.Circuit.SetActiveElement(element_name)
    currents_mag_ang = dss.CktElement.CurrentsMagAng()

    if not currents_mag_ang or len(currents_mag_ang) < 6:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    offset = 0 if terminal == 1 else 6
    if len(currents_mag_ang) < offset + 6:
        offset = 0

    mags = currents_mag_ang[offset : offset+6 : 2]
    angles = currents_mag_ang[offset+1 : offset+6 : 2]
    return list(mags), list(angles)

def extract_consumer_meter_data(meter: dict) -> dict:
    """
    Extracts electrical measurements at a given consumer meter or boundary meter from OpenDSS.
    """
    branch_id = meter.get("branch_id", meter.get("pcc_id", ""))
    bus_name = meter["bus"]

    if branch_id.startswith("transformer.") or branch_id.startswith("line."):
        element_name = branch_id
    else:
        element_name = f"line.{branch_id}"

    dss.Circuit.SetActiveElement(element_name)

    v_mags, v_angs = extract_bus_voltages(bus_name)
    v_seq = compute_symmetrical_components(v_mags, v_angs)

    terminal = 2 if branch_id.startswith("transformer.") else 1
    i_mags, i_angs = extract_element_currents(element_name, terminal=terminal)
    i_seq = compute_symmetrical_components(i_mags, i_angs)

    powers = dss.CktElement.Powers()
    offset = 6 * (terminal - 1)
    if powers and len(powers) >= offset + 6:
        p_total = sum(powers[offset : offset+6 : 2])
        q_total = sum(powers[offset+1 : offset+6 : 2])
    else:
        p_total = 0.0
        q_total = 0.0

    s_total = float(np.sqrt(p_total**2 + q_total**2))
    pf = (p_total / (s_total + 1e-6)) if s_total > 0 else 1.0

    return {
        "v_mags": v_mags,
        "v_angs": v_angs,
        "v_pos_mag": v_seq["positive"][0],
        "v_pos_ang": v_seq["positive"][1],
        "v_neg_mag": v_seq["negative"][0],
        "v_zero_mag": v_seq["zero"][0],
        "v_unbalance_pct": (v_seq["negative"][0] / (v_seq["positive"][0] + 1e-6)) * 100.0,

        "i_mags": i_mags,
        "i_angs": i_angs,
        "i_pos_mag": i_seq["positive"][0],
        "i_pos_ang": i_seq["positive"][1],
        "i_neg_mag": i_seq["negative"][0],
        "i_zero_mag": i_seq["zero"][0],
        "i_unbalance_pct": (i_seq["negative"][0] / (i_seq["positive"][0] + 1e-6)) * 100.0,

        "p_kw": p_total,
        "q_kvar": q_total,
        "s_kva": s_total,
        "pf": pf
    }

# Alias for backward compatibility
extract_pcc_data = extract_consumer_meter_data

def get_consumer_measurements(metered_consumers: list[dict]) -> dict:
    """
    Extracts electrical measurements from OpenDSS for the metered consumer and boundary nodes.
    """
    measurements = {}
    for meter in metered_consumers:
        m_id = meter.get("meter_id", meter.get("pcc_id"))
        data = extract_consumer_meter_data(meter)
        freq = float(dss.Solution.Frequency())
        data["frequency_hz"] = freq
        measurements[m_id] = data
    return measurements

# Alias for backward compatibility
get_pcc_measurements = get_consumer_measurements
