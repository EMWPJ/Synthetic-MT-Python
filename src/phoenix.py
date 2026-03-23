"""
Phoenix格式时间序列读写模块

基于Phoenix MTU-5A数据格式
参考: D:\南科大研究\数据处理研究\deal_tsn.py
"""

import struct
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional


@dataclass
class TagInfo:
    """Tag信息数据结构"""
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


class TsnFile:
    """
    Phoenix TSn文件读写类
    
    支持TS2/TS3/TS4/TS5格式
    """
    
    @staticmethod
    def _mask_data(data: np.ndarray) -> np.ndarray:
        """3字节符号扩展"""
        mask = data >= 2**23
        data = data.copy()
        data[mask] -= 2**24
        return data.astype(np.int32)
    
    @staticmethod
    def _unmask_data(data: np.ndarray) -> np.ndarray:
        """3字节无符号转换"""
        data = data.copy()
        mask = data < 0
        data[mask] += 2**24
        return data.astype(np.uint32)
    
    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取TSn文件
        
        Returns:
            data: (n, 5) 五通道数据, dtype=int32, 顺序为 [Ex, Ey, Hx, Hy, Hz]
            tags: (m, 32) tag字节数据
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
            return TsnFile._mask_data(ch)
        
        ex, ey, hx, hy, hz = [read_channel(i) for i in range(5)]
        
        data = np.stack([ex, ey, hx, hy, hz], axis=2).reshape(-1, 5)
        return data, tag_bytes
    
    @staticmethod
    def save(path: str, data: np.ndarray, tags: np.ndarray) -> None:
        """
        保存TSn文件
        
        Parameters:
            data: (n, 5) 五通道数据
            tags: (m, 32) tag字节数据
        """
        scans = int(tags[0, 10]) | (int(tags[0, 11]) << 8)
        channel = int(tags[0, 12])
        
        n_records = len(data) // scans
        data = data[:n_records * scans]
        
        ex, ey, hx, hy, hz = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        
        def write_channel(ch):
            ch = ch.reshape(n_records, scans)
            ch = TsnFile._unmask_data(ch)
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
        """
        解析tag信息
        
        Parameters:
            tag_bytes: (n, 32) tag字节数据
            time_fmt: 时间格式字符串
        
        Returns:
            tags: 结构化数组
            time_labels: 时间字符串数组
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


class TblFile:
    """
    Phoenix TBL配置文件读写类
    """
    
    TYPE_MAP = {0: 'l', 1: 'd', 2: '9s', 5: '7B'}
    
    def __init__(self, path: str = None):
        self.info = {}
        self.info_type = {}
        if path:
            self.load(path)
    
    def load(self, path: str) -> None:
        """加载TBL文件"""
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
        """保存TBL文件"""
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
        return self.info.get(key)
    
    def __setitem__(self, key: str, value):
        self.info[key] = value
    
    def keys(self):
        return self.info.keys()
