"""Phoenix TBL configuration file I/O - infrastructure layer.

TBL files contain configuration data for Phoenix MTU-5A equipment including
survey metadata, acquisition parameters, and timing information.
"""

import struct
from typing import Optional

import numpy as np


class TblFile:
    """Reader and writer for Phoenix TBL configuration files.
    
    TBL files store configuration parameters in a 25-byte row format:
    - Bytes 0-4: Parameter name (5 characters)
    - Byte 11: Data type (0=long, 1=double, 2=string, 5=time)
    - Bytes 12-24: Data value
    
    Example:
        >>> tbl = TblFile('config.tbl')
        >>> print(tbl['StationName'])
        >>> tbl['SampleRate'] = 2400
        >>> tbl.save('output.tbl')
    """
    
    TYPE_MAP = {0: 'l', 1: 'd', 2: '9s', 5: '7B'}
    
    def __init__(self, path: Optional[str] = None):
        """Initialize TblFile, optionally loading from path.
        
        Args:
            path: Optional path to TBL file to load
        """
        self.info = {}
        self.info_type = {}
        if path:
            self.load(path)
    
    def load(self, path: str) -> None:
        """Load TBL file.
        
        Args:
            path: Path to TBL file
        """
        data = np.fromfile(path, dtype=np.uint8).reshape(-1, 25)
        
        for row in data:
            name = row[:5].tobytes().split(b'\x00')[0].decode('utf-8').strip('\x00')
            dtype = row[11]
            
            self.info_type[name] = dtype
            
            if dtype == 0:  # long
                self.info[name] = struct.unpack('l', row[12:16])[0]
            elif dtype == 1:  # double
                self.info[name] = struct.unpack('d', row[12:20])[0]
            elif dtype == 2:  # string
                self.info[name] = row[12:21].tobytes().decode('utf-8').strip('\x00')
            elif dtype == 5:  # time
                self.info[name] = (row[19], row[17], row[16], row[15], 
                                 row[14], row[13], row[12], row[18])
    
    def save(self, path: str) -> None:
        """Save TBL file.
        
        Args:
            path: Output path
        """
        data = np.zeros((len(self.info), 25), dtype=np.uint8)
        
        for i, (name, dtype) in enumerate(self.info_type.items()):
            name_bytes = name.encode('utf-8')[:5].ljust(5, b'\x00')
            data[i, :5] = np.frombuffer(name_bytes, dtype=np.uint8)
            data[i, 11] = dtype
            
            if dtype == 0:
                data[i, 12:16] = np.frombuffer(struct.pack('l', self.info[name]), dtype=np.uint8)
            elif dtype == 1:
                data[i, 12:20] = np.frombuffer(struct.pack('d', self.info[name]), dtype=np.uint8)
            elif dtype == 2:
                val = self.info[name].encode('utf-8').ljust(9, b'\x00')[:9]
                data[i, 12:21] = np.frombuffer(val, dtype=np.uint8)
        
        data.tofile(path)
    
    def __getitem__(self, key: str):
        """Get configuration value by key."""
        return self.info.get(key)
    
    def __setitem__(self, key: str, value) -> None:
        """Set configuration value by key."""
        self.info[key] = value
    
    def keys(self):
        """Return configuration parameter names."""
        return self.info.keys()


def load_tbl_file(filepath: str) -> TblFile:
    """Convenience function to load TBL file.
    
    Args:
        filepath: Path to TBL file
        
    Returns:
        TblFile instance with loaded data
    """
    return TblFile(filepath)
