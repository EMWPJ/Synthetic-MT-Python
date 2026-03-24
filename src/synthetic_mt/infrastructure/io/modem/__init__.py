"""ModEM format I/O handler.

Handles reading and writing ModEM format MT data files, including the
standard ModEM binary and ASCII formats for magnetotelluric impedance data.
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
