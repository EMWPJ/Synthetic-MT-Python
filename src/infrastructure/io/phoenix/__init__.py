"""Phoenix I/O infrastructure package.

Provides infrastructure layer file I/O for Phoenix Geophysics MTU-5A format:
- TSn: Time series data (Ex, Ey, Hx, Hy, Hz)
- TBL: Configuration and metadata
"""

from .types import (
    TagInfo,
    TSN_TAG_SIZE,
    TSN_NUM_CHANNELS,
    TSN_CHANNEL_DATA_BYTES,
    TBL_ROW_SIZE,
    TBL_NAME_LENGTH,
    TBL_TYPE_MAP,
    parse_tag_bytes,
)

from .tsn import (
    TsnFile,
    load_tsn_file,
)

from .tbl import (
    TblFile,
    load_tbl_file,
)

__all__ = [
    # Types
    'TagInfo',
    'TSN_TAG_SIZE',
    'TSN_NUM_CHANNELS',
    'TSN_CHANNEL_DATA_BYTES',
    'TBL_ROW_SIZE',
    'TBL_NAME_LENGTH',
    'TBL_TYPE_MAP',
    'parse_tag_bytes',
    # TSn
    'TsnFile',
    'load_tsn_file',
    # TBL
    'TblFile',
    'load_tbl_file',
]
