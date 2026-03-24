"""Phoenix format types and data structures.

Phoenix is a data format used by Phoenix Geophysics MTU-5A equipment.
This module defines the types used in TSn (time series) and TBL (configuration) files.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple
import numpy as np


@dataclass
class TagInfo:
    """Tag information data structure for TSn time series files.
    
    Each scan in a Phoenix TSn file has a 32-byte tag containing:
    - Timestamp (year, month, day, hour, minute, second)
    - Serial number
    - Scan count
    - Channel count
    - Sample rate
    - Clock status and error
    """
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    serial_number: int
    scans: int
    channel: int
    sample_rate: int
    clock_status: int
    clock_error: int


# Phoenix TSn file constants
TSN_TAG_SIZE = 32  # bytes per tag record
TSN_NUM_CHANNELS = 5  # Ex, Ey, Hx, Hy, Hz
TSN_CHANNEL_DATA_BYTES = 3  # 3 bytes per sample per channel

# TBL file constants  
TBL_ROW_SIZE = 25  # bytes per row
TBL_NAME_LENGTH = 5
TBL_TYPE_MAP = {0: 'l', 1: 'd', 2: '9s', 5: '7B'}


def parse_tag_bytes(tag_bytes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Parse tag bytes into structured array and time labels.
    
    Args:
        tag_bytes: (n, 32) tag byte data
        
    Returns:
        tags: structured numpy array with parsed fields
        time_labels: datetime array with timestamps
    """
    if len(tag_bytes.shape) == 1:
        tag_bytes = tag_bytes.reshape(1, -1)
    
    dtype = np.dtype([
        ('year', 'i2'), ('month', 'i1'), ('day', 'i1'),
        ('hour', 'i1'), ('minute', 'i1'), ('second', 'i1'),
        ('serial_number', 'i2'), ('scans', 'i2'), ('channel', 'i1'),
        ('sample_len', 'i1'), ('a_flag', 'i1'), ('b_flag', 'i1'),
        ('c_flag', 'i1'), ('sample_rate', 'i2'), ('sample_unit', 'i1'),
        ('clock_status', 'i1'), ('clock_error', 'i4')
    ])
    
    tags = np.empty(len(tag_bytes), dtype=dtype)
    
    tags['year'] = tag_bytes[:, 7].astype('i2') * 100 + tag_bytes[:, 5]
    tags['month'] = tag_bytes[:, 4]
    tags['day'] = tag_bytes[:, 3]
    tags['hour'] = tag_bytes[:, 2]
    tags['minute'] = tag_bytes[:, 1]
    tags['second'] = tag_bytes[:, 0]
    tags['serial_number'] = tag_bytes[:, 8].astype('i2') | (tag_bytes[:, 9].astype('i2') << 8)
    tags['scans'] = tag_bytes[:, 10].astype('i2') | (tag_bytes[:, 11].astype('i2') << 8)
    tags['channel'] = tag_bytes[:, 12]
    tags['sample_rate'] = tag_bytes[:, 18].astype('i2') | (tag_bytes[:, 19].astype('i2') << 8)
    
    time_labels = np.array([
        datetime(int(tags[i]['year']), int(tags[i]['month']), int(tags[i]['day']),
                int(tags[i]['hour']), int(tags[i]['minute']), int(tags[i]['second']))
        for i in range(len(tags))
    ])
    
    return tags, time_labels
