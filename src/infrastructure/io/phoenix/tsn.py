"""Phoenix TSn time series file I/O - infrastructure layer.

TSn (Time Series n) files contain time series data from Phoenix MTU-5A equipment.
Supported formats: TS2/TS3/TS4/TS5.
"""

from pathlib import Path
from typing import Tuple

import numpy as np

from .types import parse_tag_bytes, TSN_TAG_SIZE


class TsnFile:
    """Reader and writer for Phoenix TSn time series files.
    
    TSn files store 5-channel electromagnetic data (Ex, Ey, Hx, Hy, Hz)
    with 32-byte tag records containing timing and status information.
    
    Example:
        >>> data, tags = TsnFile.load('measurement.tsn')
        >>> print(f"Loaded {len(data)} samples")
        >>> TsnFile.save('output.tsn', data, tags)
    """
    
    @staticmethod
    def _mask_data(data: np.ndarray) -> np.ndarray:
        """3-byte sign extension for Phoenix data format.
        
        Phoenix uses 3-byte signed integers that need sign extension
        to proper 4-byte integers.
        """
        mask = data >= 2**23
        data = data.copy()
        data[mask] -= 2**24
        return data.astype(np.int32)
    
    @staticmethod
    def _unmask_data(data: np.ndarray) -> np.ndarray:
        """Convert 3-byte signed integers to unsigned.
        
        Reverses the sign extension operation.
        """
        data = data.copy()
        mask = data < 0
        data[mask] += 2**24
        return data.astype(np.uint32)
    
    @classmethod
    def load(cls, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read TSn file.
        
        Args:
            path: Path to TSn file
            
        Returns:
            data: (n, 5) five-channel data, dtype=int32, order [Ex, Ey, Hx, Hy, Hz]
            tags: (m, 32) tag byte data
        """
        with open(path, 'rb') as f:
            tsn_bytes = np.fromfile(f, dtype=np.uint8)
        
        scans = tsn_bytes[10] | (tsn_bytes[11] << 8)
        channel = tsn_bytes[12]
        len_tag = tsn_bytes[13]
        
        tsn_bytes = tsn_bytes.reshape(-1, len_tag + scans * channel * 3)
        tag_bytes = tsn_bytes[:, :32]
        data_bytes = tsn_bytes[:, 32:].reshape(tsn_bytes.shape[0], scans, channel * 3)
        
        def read_channel(idx):
            ch = data_bytes[:, :, idx::3].astype(np.uint32)
            ch = (ch | (data_bytes[:, :, idx+1:idx+3:3].astype(np.uint32) << 8) |
                  (data_bytes[:, :, idx+2:idx+3:3].astype(np.uint32) << 16))
            return cls._mask_data(ch)
        
        ex, ey, hx, hy, hz = [read_channel(i) for i in range(5)]
        
        data = np.stack([ex, ey, hx, hy, hz], axis=2).reshape(-1, 5)
        return data, tag_bytes
    
    @classmethod
    def save(cls, path: str, data: np.ndarray, tags: np.ndarray) -> None:
        """Save TSn file.
        
        Args:
            path: Output path
            data: (n, 5) five-channel data
            tags: (m, 32) tag byte data
        """
        scans = int(tags[0, 10]) | (int(tags[0, 11]) << 8)
        channel = int(tags[0, 12])
        
        n_records = len(data) // scans
        data = data[:n_records * scans]
        
        ex, ey, hx, hy, hz = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        
        def write_channel(ch):
            ch = ch.reshape(n_records, scans)
            ch = cls._unmask_data(ch)
            b0 = ch & 0xFF
            b1 = (ch >> 8) & 0xFF
            b2 = (ch >> 16) & 0xFF
            return np.concatenate([b0, b1, b2], axis=1)
        
        data_bytes = np.concatenate([
            write_channel(ex), write_channel(ey), write_channel(hx),
            write_channel(hy), write_channel(hz)
        ], axis=1)
        
        tsn_bytes = np.concatenate([tags[:n_records], data_bytes], axis=1).reshape(-1)
        with open(path, 'wb') as f:
            f.write(tsn_bytes.astype(np.uint8).tobytes())
    
    @staticmethod
    def parse_tags(tag_bytes: np.ndarray, 
                   time_fmt: str = "%y-%m-%d %H:%M:%S") -> Tuple[np.ndarray, np.ndarray]:
        """Parse tag information.
        
        Args:
            tag_bytes: (n, 32) tag byte data
            time_fmt: Time format string (unused, kept for API compatibility)
            
        Returns:
            tags: Structured numpy array with parsed fields
            time_labels: Datetime array with timestamps
        """
        return parse_tag_bytes(tag_bytes)


def load_tsn_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to load TSn file.
    
    Args:
        filepath: Path to TSn file
        
    Returns:
        data: (n, 5) five-channel data
        tags: (m, 32) tag byte data
    """
    return TsnFile.load(filepath)
