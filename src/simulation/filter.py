import numpy as np
from scipy import signal


def apply_bandpass_filter(
    data: np.ndarray,
    fundamental_hz: float,
    bandwidth_hz: float,
    fs: float,
    order: int,
) -> np.ndarray:
    """
    Applies a SciPy Butterworth band-pass filter using second-order sections (SOS)
    to isolate the primary fundamental frequency and remove both lower frequency components
    (DC / low-frequency) and higher order harmonics.

    Parameters:
        data: 1D or 2D numpy array containing time-series signals.
        fundamental_hz: Target primary fundamental frequency in Hz (e.g. 50.0 or 60.0).
        bandwidth_hz: Passband width around fundamental frequency in Hz (e.g. 10.0 Hz for +/- 5 Hz).
        fs: Sampling frequency in Hz (e.g. 10,000 Hz).
        order: Filter order.

    Returns:
        Filtered signal array containing only the primary fundamental frequency component.
    """
    if fundamental_hz <= 0.0:
        raise ValueError(f"fundamental_hz must be positive, got {fundamental_hz}")
    if bandwidth_hz <= 0.0:
        raise ValueError(f"bandwidth_hz must be positive, got {bandwidth_hz}")
    if fs <= 0.0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}")
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}")

    lowcut = fundamental_hz - (bandwidth_hz / 2.0)
    highcut = fundamental_hz + (bandwidth_hz / 2.0)

    if lowcut <= 0.0:
        raise ValueError(f"Filter lowcut frequency ({lowcut} Hz) must be positive")
    nyquist = 0.5 * fs
    if highcut >= nyquist:
        raise ValueError(
            f"Filter highcut frequency ({highcut} Hz) exceeds Nyquist frequency ({nyquist} Hz)"
        )

    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(order, [low, high], btype="bandpass", analog=False, output="sos")

    data_arr = np.asarray(data)
    if data_arr.ndim == 1:
        return signal.sosfiltfilt(sos, data_arr)
    elif data_arr.ndim == 2:
        filtered = np.zeros_like(data_arr)
        for col in range(data_arr.shape[1]):
            filtered[:, col] = signal.sosfiltfilt(sos, data_arr[:, col])
        return filtered
    else:
        return data_arr


def remove_low_frequency_components(
    data: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int,
) -> np.ndarray:
    """
    Adapter function delegating to apply_bandpass_filter to isolate the primary fundamental frequency
    and remove higher or lower order harmonics.
    """
    fundamental_hz = 50.0 if cutoff_hz >= 100.0 else cutoff_hz
    return apply_bandpass_filter(
        data=data,
        fundamental_hz=fundamental_hz,
        bandwidth_hz=10.0,
        fs=fs,
        order=order,
    )
