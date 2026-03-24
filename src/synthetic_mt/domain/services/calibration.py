"""Calibration service - System calibration and response handling.

This module contains calibration-related domain logic for Phoenix MTU-5A systems.
"""

import numpy as np
import struct
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CalibrationData:
    """标定数据"""
    channel: int
    serial_number: int
    sample_rate: int
    frequency: float
    amplitude_re: float
    amplitude_im: float
    phase_re: float
    phase_im: float


class ClbFile:
    """
    Phoenix CLB标定文件读写类
    
    CLB文件格式: 二进制格式, 包含通道标定响应参数
    """
    
    CAL_CHANNELS = {0: 'Ex', 1: 'Ey', 2: 'Hx', 3: 'Hy', 4: 'Hz'}
    
    def __init__(self, path: Optional[str] = None):
        self.calibrations: List[CalibrationData] = []
        if path:
            self.load(path)
    
    def load(self, path: str) -> None:
        """加载CLB标定文件"""
        with open(path, 'rb') as f:
            data = f.read()
        
        self.calibrations = []
        offset = 0
        
        while offset < len(data) - 64:
            try:
                channel = data[offset]
                serial = int.from_bytes(data[offset+1:offset+3], 'little')
                sample_rate = int.from_bytes(data[offset+3:offset+5], 'little')
                frequency = struct.unpack('f', data[offset+5:offset+9])[0]
                amp_re = struct.unpack('f', data[offset+9:offset+13])[0]
                amp_im = struct.unpack('f', data[offset+13:offset+17])[0]
                phase_re = struct.unpack('f', data[offset+17:offset+21])[0]
                phase_im = struct.unpack('f', data[offset+21:offset+25])[0]
                
                cal = CalibrationData(
                    channel=channel,
                    serial_number=serial,
                    sample_rate=sample_rate,
                    frequency=frequency,
                    amplitude_re=amp_re,
                    amplitude_im=amp_im,
                    phase_re=phase_re,
                    phase_im=phase_im,
                )
                self.calibrations.append(cal)
                offset += 64
            except Exception:
                break
    
    def save(self, path: str) -> None:
        """保存CLB标定文件"""
        with open(path, 'wb') as f:
            for cal in self.calibrations:
                f.write(bytes([cal.channel]))
                f.write(cal.serial_number.to_bytes(2, 'little'))
                f.write(cal.sample_rate.to_bytes(2, 'little'))
                f.write(struct.pack('f', cal.frequency))
                f.write(struct.pack('f', cal.amplitude_re))
                f.write(struct.pack('f', cal.amplitude_im))
                f.write(struct.pack('f', cal.phase_re))
                f.write(struct.pack('f', cal.phase_im))
                f.write(b'\x00' * 39)
    
    def get_calibration(self, channel: int, frequency: float) -> Optional[CalibrationData]:
        """获取指定通道和频率的标定数据"""
        best_match = None
        best_diff = float('inf')
        
        for cal in self.calibrations:
            if cal.channel == channel:
                diff = abs(cal.frequency - frequency)
                if diff < best_diff:
                    best_diff = diff
                    best_match = cal
        
        return best_match
    
    def apply_calibration(self, data: np.ndarray, channel: int,
                         sample_rate: float) -> np.ndarray:
        """
        应用标定校正到时间序列
        
        Parameters:
            data: 时间序列数据
            channel: 通道号 (0-4)
            sample_rate: 采样率
        
        Returns:
            校正后的时间序列
        """
        freqs = np.fft.rfftfreq(len(data), 1.0/sample_rate)
        fft_data = np.fft.rfft(data)
        
        for i, f in enumerate(freqs):
            if f == 0:
                continue
            cal = self.get_calibration(channel, f)
            if cal:
                resp = complex(cal.amplitude_re, cal.amplitude_im) * \
                       np.exp(1j * complex(cal.phase_re, cal.phase_im))
                if abs(resp) > 1e-10:
                    fft_data[i] /= resp
        
        return np.fft.irfft(fft_data, n=len(data))


class ClcFile:
    """
    Phoenix CLC标定配置文件读写类
    
    CLC文件是ASCII格式的标定配置文件
    """
    
    def __init__(self, path: Optional[str] = None):
        self.info = {}
        if path:
            self.load(path)
    
    def load(self, path: str) -> None:
        """加载CLC标定配置文件"""
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    key = parts[0]
                    try:
                        if parts[1] == 'd':
                            self.info[key] = float(parts[2])
                        elif parts[1] == 'i':
                            self.info[key] = int(parts[2])
                        else:
                            self.info[key] = ' '.join(parts[2:])
                    except (ValueError, IndexError):
                        pass
    
    def save(self, path: str) -> None:
        """保存CLC标定配置文件"""
        with open(path, 'w') as f:
            f.write('# Phoenix CLC Calibration File\n\n')
            for key, value in self.info.items():
                if isinstance(value, float):
                    f.write(f'{key} d {value}\n')
                elif isinstance(value, int):
                    f.write(f'{key} i {value}\n')
                else:
                    f.write(f'{key} s {value}\n')
    
    def __getitem__(self, key: str):
        return self.info.get(key)
    
    def __setitem__(self, key: str, value):
        self.info[key] = value


class SystemCalibrator:
    """
    系统标定器 - 调用SysCal V7进行仪器响应标定
    
    用于Phoenix MTU-5A系统的现场标定
    """
    
    def __init__(self, syscal_exe_path: str = './SysCal_V7.exe',
                 tmp_dir: str = './tmp'):
        self.syscal_exe_path = syscal_exe_path
        self.tmp_dir = tmp_dir
        self.responses: Optional[np.ndarray] = None
    
    def run_calibration(self, tbl_config, 
                        config_dir: str = './Config') -> bool:
        """
        运行SysCal V7标定程序
        
        Parameters:
            tbl_config: TBL配置文件
            config_dir: 配置文件目录
        
        Returns:
            是否成功
        """
        import subprocess
        from pathlib import Path
        
        tmp_path = Path(self.tmp_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        
        try:
            result = subprocess.run(
                [self.syscal_exe_path],
                cwd=config_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return result.returncode == 0
        except Exception:
            return False
    
    def get_system_responses(self, frequencies: np.ndarray,
                            channel_ex: int, channel_ey: int,
                            channel_hx: int, channel_hy: int,
                            channel_hz: int) -> np.ndarray:
        """
        获取系统响应
        
        Parameters:
            frequencies: 频率数组
            channel_*: 各通道索引
        
        Returns:
            系统响应数组 (5 x n_frequencies)
        """
        n = len(frequencies)
        responses = np.ones((5, n), dtype=complex)
        
        responses[0] = 1e-6
        responses[1] = 1e-6
        responses[2] = 1e-9
        responses[3] = 1e-9
        responses[4] = 1e-9
        
        return responses
    
    def apply_calibration(self, data: np.ndarray, 
                         channel: int, frequencies: np.ndarray) -> np.ndarray:
        """
        应用系统标定到时间序列
        
        Parameters:
            data: 时间序列数据
            channel: 通道号 (0-4)
            frequencies: 频率数组
        
        Returns:
            标定后的时间序列
        """
        if self.responses is None:
            return data
        
        fft_data = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data))
        
        for i, f in enumerate(freqs):
            if f > 0 and i < len(self.responses[channel]):
                resp = self.responses[channel, i]
                if abs(resp) > 1e-10:
                    fft_data[i] /= resp
        
        return np.fft.irfft(fft_data, n=len(data))
