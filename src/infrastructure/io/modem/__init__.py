"""ModEM I/O infrastructure package.

Provides infrastructure layer file I/O for ModEM magnetotelluric format.
"""

from .types import (
    ModEMBlockType,
    ModEMHeader,
    ModEMImpedanceData,
    ModEMTipperData,
    ModEMFieldData,
    MODEM_SCALE_IMPEDANCE,
    MODEM_SCALE_EFIELD,
    MODEM_SCALE_HFIELD,
)

from .reader import (
    ModEMReader,
    load_modem_file,
)

__all__ = [
    'ModEMBlockType',
    'ModEMHeader',
    'ModEMImpedanceData',
    'ModEMTipperData',
    'ModEMFieldData',
    'MODEM_SCALE_IMPEDANCE',
    'MODEM_SCALE_EFIELD',
    'MODEM_SCALE_HFIELD',
    'ModEMReader',
    'load_modem_file',
]
