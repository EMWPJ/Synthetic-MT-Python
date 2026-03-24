"""ModEM format types and enums.

ModEM (Modified ModEM) is a magnetotelluric forward modeling format.
It contains impedance, tipper, and electromagnetic field data.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List


class ModEMBlockType(Enum):
    """ModEM data block types."""
    IMPEDANCE = "Full_Impedance"
    TIPPER = "Full_Vertical_Components"
    EM_FIELDS = "EM_Fields"
    UNKNOWN = "Unknown"


@dataclass
class ModEMHeader:
    """ModEM block header parsed data."""
    block_type: ModEMBlockType
    frequency_count: int
    site_count: int = 1
    extra: str = ""


@dataclass
class ModEMImpedanceData:
    """Impedance tensor data for one frequency."""
    freq: float
    zxx: complex
    zxy: complex
    zyx: complex
    zyy: complex


@dataclass
class ModEMTipperData:
    """Tipper (vertical magnetic transfer function) data."""
    freq: float
    tzx: complex
    tzy: complex


@dataclass 
class ModEMFieldData:
    """EM field components for one frequency."""
    freq: float
    ex1: complex
    ey1: complex
    hx1: complex
    hy1: complex
    hz1: complex
    ex2: complex
    ey2: complex
    hx2: complex
    hy2: complex
    hz2: complex


# Unit conversion constants used in ModEM format
MODEM_SCALE_IMPEDANCE = 1e6 / 4e2 / 3.141592653589793
MODEM_SCALE_EFIELD = 4e2 * 3.141592653589793  # [mv/km] -> [V/m]
MODEM_SCALE_HFIELD = 1e9  # [nT] -> [A/m]
