import numpy as np
from scipy import signal


def remove_low_frequency_components(data: np.ndarray, cutoff_hz: float = 100.0, fs: float = 10000.0, order: int = 4) -> np.ndarray:
    """
    Removes low-frequency fundamental/dc components from time-series signal arrays
    using a SciPy Butterworth high-pass filter (scipy.signal).

    Parameters:
        data: 1D or 2D numpy array containing time-series signals.
        cutoff_hz: High-pass cutoff frequency in Hz (default: 100 Hz to remove 50 Hz fundamental).
        fs: Sampling frequency in Hz (default: 10,000 Hz).
        order: Filter order (default: 4).

    Returns:
        Filtered signal array of the same shape as input.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

    data_arr = np.asarray(data)
    if data_arr.ndim == 1:
        return signal.filtfilt(b, a, data_arr)
    elif data_arr.ndim == 2:
        # Filter each phase / channel along axis 0
        filtered = np.zeros_like(data_arr)
        for col in range(data_arr.shape[1]):
            filtered[:, col] = signal.filtfilt(b, a, data_arr[:, col])
        return filtered
    else:
        return data_arr
