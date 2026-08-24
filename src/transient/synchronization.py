from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PCCMeasurement:
    pcc_id: str
    timestamp_s: float
    voltage_abc: tuple  # (v_a, v_b, v_c)
    current_abc: tuple  # (i_a, i_b, i_c)
    p_kw: float
    q_kvar: float
    s_kva: float
    pf: float
    frequency_hz: float
    rocof_hz_s: float
    v_sequence: tuple   # (v_pos, v_neg, v_zero)
    i_sequence: tuple   # (i_pos, i_neg, i_zero)

@dataclass
class SpectrumAnalyzerMeasurement:
    pcc_id: str
    timestamp_s: float
    voltage_fft_magnitudes: list
    current_fft_magnitudes: list
    wavelet_coefficients: dict
    features: dict

def synchronize_measurements(pcc_data_dict: dict, timestamp_s: float = 0.0) -> dict[str, PCCMeasurement]:
    """
    Synchronizes the steady-state measurements into canonical PCCMeasurement records.
    """
    synced = {}
    for pcc_id, data in pcc_data_dict.items():
        v_rms_abc = tuple(data["v_mags"])
        i_rms_abc = tuple(data["i_mags"])

        synced[pcc_id] = PCCMeasurement(
            pcc_id=pcc_id,
            timestamp_s=timestamp_s,
            voltage_abc=v_rms_abc,
            current_abc=i_rms_abc,
            p_kw=data["p_kw"],
            q_kvar=data["q_kvar"],
            s_kva=data["s_kva"],
            pf=data["pf"],
            frequency_hz=data.get("frequency_hz", 50.0),
            rocof_hz_s=0.0,
            v_sequence=(data["v_pos_mag"], data["v_neg_mag"], data["v_zero_mag"]),
            i_sequence=(data["i_pos_mag"], data["i_neg_mag"], data["i_zero_mag"])
        )
    return synced

def synchronize_spectrum_analyzer_measurements(processed_pccs_dict: dict, timestamp_s: float = 0.0) -> dict[str, SpectrumAnalyzerMeasurement]:
    """
    Synchronizes and aligns the spectrum analyzer (high-frequency wavelet/spectral) measurements.
    """
    synced_spectral = {}
    for pcc_id, processed in processed_pccs_dict.items():
        synced_spectral[pcc_id] = SpectrumAnalyzerMeasurement(
            pcc_id=pcc_id,
            timestamp_s=timestamp_s,
            voltage_fft_magnitudes=processed.voltage_fft.tolist(),
            current_fft_magnitudes=processed.current_fft.tolist(),
            wavelet_coefficients={
                "voltage_swt": [[[cA.tolist(), cD.tolist()] for cA, cD in p_swt] for p_swt in processed.voltage_swt],
                "current_swt": [[[cA.tolist(), cD.tolist()] for cA, cD in p_swt] for p_swt in processed.current_swt]
            },
            features=processed.features
        )
    return synced_spectral
