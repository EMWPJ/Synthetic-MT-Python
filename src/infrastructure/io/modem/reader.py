"""ModEM file reader - infrastructure I/O for ModEM format.

This module provides file I/O for the ModEM (Modified ModEM) magnetotelluric
forward modeling format. It wraps the domain parsing logic from synthesis.py.
"""

from typing import List
from pathlib import Path

from ...domain.services.synthesis import load_modem_file as domain_load_modem_file
from ...domain.entities import ForwardSite


class ModEMReader:
    """Reader for ModEM format forward modeling files.
    
    ModEM format contains:
    - Full_Impedance: Impedance tensor Zxx, Zxy, Zyx, Zyy
    - Full_Vertical_Components: Tipper data Tzx, Tzy  
    - EM_Fields: Electromagnetic field components Ex, Ey, Hx, Hy, Hz
    """
    
    def __init__(self, filepath: str):
        """Initialize reader with path to ModEM file.
        
        Args:
            filepath: Path to ModEM format file
        """
        self.filepath = Path(filepath)
    
    def read(self) -> List[ForwardSite]:
        """Read and parse ModEM file.
        
        Returns:
            List of ForwardSite objects with parsed EM data
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is not valid ModEM format
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"ModEM file not found: {self.filepath}")
        
        return domain_load_modem_file(str(self.filepath))
    
    @property
    def filename(self) -> str:
        """Return the filename."""
        return self.filepath.name


def load_modem_file(filepath: str) -> List[ForwardSite]:
    """Convenience function to load ModEM file.
    
    Args:
        filepath: Path to ModEM format file
        
    Returns:
        List of ForwardSite objects
    """
    reader = ModEMReader(filepath)
    return reader.read()
