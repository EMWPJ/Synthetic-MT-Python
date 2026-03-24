"""CSV format time series exporter."""

from typing import Optional

import numpy as np


def save_csv_timeseries(file_path: str,
                        ex: np.ndarray, ey: np.ndarray,
                        hx: np.ndarray, hy: np.ndarray, hz: np.ndarray,
                        header: Optional[str] = None) -> str:
    """
    Save time series in CSV format.

    Parameters:
        file_path: Output file path
        ex, ey, hx, hy, hz: Electromagnetic field time series
        header: CSV header comment

    Returns:
        Path to saved file
    """
    data = np.column_stack([ex, ey, hx, hy, hz])

    if header is None:
        header = "Ex(V/m),Ey(V/m),Hx(A/m),Hy(A/m),Hz(A/m)"

    np.savetxt(file_path, data, delimiter=',', header=header)

    return file_path
