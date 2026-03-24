"""NumPy format time series exporter and loader."""

from typing import Tuple

import numpy as np


def save_numpy_timeseries(file_path: str,
                          ex: np.ndarray, ey: np.ndarray,
                          hx: np.ndarray, hy: np.ndarray, hz: np.ndarray) -> str:
    """
    Save time series in NumPy format.

    Parameters:
        file_path: Output file path
        ex, ey, hx, hy, hz: Electromagnetic field time series

    Returns:
        Path to saved file
    """
    data = np.column_stack([ex, ey, hx, hy, hz])
    np.save(file_path, data)

    return file_path


def load_numpy_timeseries(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load time series from NumPy format.

    Parameters:
        file_path: File path

    Returns:
        Tuple of (ex, ey, hx, hy, hz) arrays
    """
    data = np.load(file_path)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
