"""GMT format time series exporter."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def save_gmt_timeseries(dir_path: str, site_name: str,
                        ex: np.ndarray, ey: np.ndarray,
                        hx: np.ndarray, hy: np.ndarray, hz: np.ndarray,
                        begin_time: datetime, sample_rate: float,
                        unit: str = 'V/m') -> str:
    """
    Save time series in GMT-compatible text format.

    GMT (Generic Mapping Tools) compatible time series format.
    Each line: Year Month Day Hour Minute Second Ex Ey Hx Hy Hz

    Parameters:
        dir_path: Output directory
        site_name: Site name
        ex, ey, hx, hy, hz: Electromagnetic field time series
        begin_time: Start time
        sample_rate: Sampling rate (Hz)
        unit: Data unit ('V/m' for E, 'A/m' for H)

    Returns:
        Path to saved file
    """
    output_dir = Path(dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"{site_name}.txt"

    n_samples = len(ex)
    dt = 1.0 / sample_rate

    with open(file_path, 'w') as f:
        f.write("# GMT时间序列数据\n")
        f.write(f"# 测点: {site_name}\n")
        f.write(f"# 开始时间: {begin_time.isoformat()}\n")
        f.write(f"# 采样率: {sample_rate} Hz\n")
        f.write(f"# 单位: Ex Ey [V/m], Hx Hy Hz [A/m]\n")
        f.write("# Year Month Day Hour Minute Second Ex Ey Hx Hy Hz\n")

        current_time = begin_time

        for i in range(n_samples):
            if i > 0 and i % 100000 == 0:
                current_time = begin_time.replace(
                    second=begin_time.second + int(i / sample_rate)
                )

            f.write(
                f"{current_time.year:4d} "
                f"{current_time.month:02d} "
                f"{current_time.day:02d} "
                f"{current_time.hour:02d} "
                f"{current_time.minute:02d} "
                f"{current_time.second:02d} "
                f"{ex[i]:15.6e} "
                f"{ey[i]:15.6e} "
                f"{hx[i]:15.6e} "
                f"{hy[i]:15.6e} "
                f"{hz[i]:15.6e}\n"
            )

            current_time = datetime.fromtimestamp(
                begin_time.timestamp() + i * dt
            )

    return str(file_path)
