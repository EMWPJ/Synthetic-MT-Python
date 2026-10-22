# -*- coding: utf-8 -*-
"""Phoenix TBL 二进制读写（MTDataPro TTBLPackedRecord 逐位等价移植）。

格式（packed 记录 25 字节，typed file 流）:
    Name[0..3]  AnsiChar  4B
    Null        Byte      1B
    MTUShort    Smallint  2B LE
    MTULong     Longint   4B LE
    Identifier  Byte      1B  (0=int32, 1=double, 2=str≤9 零终止, 3=str8 定长, 4=str≤13 零终止, 5=datetime)
    Data[0..12] 13B

datetime 编码: [0]=秒 [1]=分 [2]=时 [3]=日 [4]=月 [5]=年%100 [6]=DayOfWeek-1 [7]=世纪
（Data[7]=0 视为无效零日期）
文件尾: Name[0]=#03 的终止记录（读取时跳过一切 name[0]=3 的记录）。
"""
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta

REC_SIZE = 25
TERMINATOR = b'\x03\x00\x00\x00' + b'\x00' * 21

# Delphi TDateTime 纪元: 1899-12-30
DELPHI_EPOCH = datetime(1899, 12, 30)


def to_tdatetime(dt: datetime) -> float:
    """datetime → Delphi TDateTime（自 1899-12-30 的天数）。

    与 Delphi EncodeDateTime 同式：整数天 + 毫秒整数/86400000——
    若用 total_seconds()/86400 一次大除会差 1ULP，经毫秒整数量化
    （_phase_setup 的 deltatime）后放大为 1ms 相位差（RSLPW 对拍定位）。"""
    days = (dt.date() - DELPHI_EPOCH.date()).days
    ms = dt.hour * 3600000 + dt.minute * 60000 + dt.second * 1000
    return days + ms / 86400000.0


def from_tdatetime(value: float) -> datetime:
    """Delphi TDateTime → datetime（毫秒精度截断，Delphi 编码本身只到秒）。"""
    return DELPHI_EPOCH + timedelta(seconds=round(value * 86400.0))


@dataclass
class TBLRecord:
    raw: bytes                                   # 完整 25B 原始记录
    name: str = ''
    identifier: int = 0

    @property
    def value(self):
        """按 Identifier 解码（对应 TTBLPackedRecord.Value）。"""
        data = self.raw[12:25]
        ident = self.raw[11]
        if ident == 0:
            return struct.unpack('<i', data[0:4])[0]
        if ident == 1:
            return struct.unpack('<d', data[0:8])[0]
        if ident == 2:
            s = data.split(b'\x00', 1)[0]
            return s.decode('latin-1')
        if ident == 3:
            return data[0:8].decode('latin-1')
        if ident == 4:
            s = data.split(b'\x00', 1)[0]
            return s.decode('latin-1')
        if ident == 5:
            if data[7] == 0:
                return None
            year = data[7] * 100 + data[5]
            return datetime(year, data[4], data[3], data[2], data[1], data[0])
        return None

    def set_string_value(self, text: str):
        """字符串 setter（Identifier 2/3/4 通用编码：逐字节 + 零填充到 13B）。"""
        raw = bytearray(self.raw)
        b = text.encode('latin-1')
        data = b + b'\x00' * (13 - len(b))
        raw[12:25] = data[:13]
        self.raw = bytes(raw)

    def set_datetime_value(self, dt: datetime):
        """datetime setter（Identifier=5：[秒,分,时,日,月,年%100,周日码,世纪]）。"""
        dow = (dt.weekday() + 1) % 7          # Delphi DayOfWeek-1（周日=0）
        raw = bytearray(self.raw)
        raw[11] = 5
        raw[12:25] = (bytes([dt.second, dt.minute, dt.hour, dt.day, dt.month,
                             dt.year % 100, dow, dt.year // 100])
                      + b'\x00' * 5)
        self.raw = bytes(raw)


def make_string_record(name: str, text: str, identifier: int = 4) -> TBLRecord:
    """构造一个新的字符串记录（用于输出时补充缺失字段）。"""
    b = text.encode('latin-1')
    data = b + b'\x00' * (13 - len(b))
    raw = (name.encode('ascii', 'replace')[:4].ljust(4, b'\x00') + b'\x00' +
           struct.pack('<h', 0) + struct.pack('<i', 0) +
           bytes([identifier]) + data[:13])
    return TBLRecord(raw=raw, name=name, identifier=identifier)


def make_datetime_record(name: str, dt: datetime) -> TBLRecord:
    """构造 datetime(Identifier=5) 记录：[秒,分,时,日,月,年%100,周日码,世纪]。"""
    dow = (dt.weekday() + 1) % 7          # Delphi DayOfWeek-1（周日=0）
    data = bytes([dt.second, dt.minute, dt.hour, dt.day, dt.month,
                  dt.year % 100, dow, dt.year // 100]) + b'\x00' * 5
    raw = (name.encode('ascii', 'replace')[:4].ljust(4, b'\x00') + b'\x00' +
           struct.pack('<h', 0) + struct.pack('<i', 0) +
           bytes([5]) + data[:13])
    return TBLRecord(raw=raw, name=name, identifier=5)


class TBL:
    """读取后的 TBL：记录列表 + 常用字段访问（等价 TTBLHelper 的 getter 子集）。"""

    def __init__(self):
        self.records: list[TBLRecord] = []
        self.path = ''
        self.file_stem = ''

    def _first(self, name: str) -> TBLRecord | None:
        for r in self.records:
            if r.name == name:
                return r
        return None

    def get_str(self, name: str, default: str = '') -> str:
        r = self._first(name)
        if r is None:
            return default
        v = r.value
        if isinstance(v, str):
            return v
        if isinstance(v, int):
            return str(v)          # SNUM 等数值记录按字符串读（TBLRecordOf[].Value 字符串语义）
        return default

    def get_int(self, name: str, default: int = 0) -> int:
        r = self._first(name)
        if r is None:
            return default
        v = r.value
        return v if isinstance(v, int) else default

    def get_datetime(self, name: str) -> datetime | None:
        r = self._first(name)
        if r is None:
            return None
        return r.value if isinstance(r.value, datetime) else None

    # —— CLI 所需字段（记录名与 PhoenixB TTBLHelper 一致）——
    @property
    def snum(self) -> str:
        return self.get_str('SNUM')

    @property
    def hxsn(self) -> str:
        return self.get_str('HXSN')

    @property
    def hysn(self) -> str:
        return self.get_str('HYSN')

    @property
    def hzsn(self) -> str:
        return self.get_str('HZSN')

    @property
    def ftim(self) -> datetime | None:
        return self.get_datetime('FTIM')

    @property
    def ltim(self) -> datetime | None:
        return self.get_datetime('LTIM')

    @property
    def hsmp(self) -> int:
        return self.get_int('HSMP')

    @property
    def l2ns(self) -> int:
        return self.get_int('L2NS')

    @property
    def l3ns(self) -> int:
        return self.get_int('L3NS')

    @property
    def l4ns(self) -> int:
        return self.get_int('L4NS')

    @property
    def tsfilename(self) -> str:
        return self.get_str('FILE')

    def set_str_field(self, name: str, value: str):
        """设置字符串记录值（不存在则追加；等价 TBLRecordOf[name].Value := v）。"""
        r = self._first(name)
        if r is None:
            self.records.append(make_string_record(name, value))
        else:
            r.set_string_value(value)

    def set_datetime_field(self, name: str, value: datetime):
        """设置 datetime(Identifier=5) 记录值（不存在则追加）。"""
        r = self._first(name)
        if r is None:
            self.records.append(make_datetime_record(name, value))
        else:
            r.set_datetime_value(value)

    # —— 读写 ——
    @classmethod
    def load(cls, fname: str) -> 'TBL':
        with open(fname, 'rb') as f:
            blob = f.read()
        if len(blob) % REC_SIZE != 0:
            raise ValueError(f'TBL 文件长度非 {REC_SIZE} 的整数倍: {fname}')
        tbl = cls()
        tbl.path = fname.replace('\\', '/')
        stem = tbl.path.rsplit('/', 1)[-1]
        tbl.file_stem = stem.rsplit('.', 1)[0]
        for off in range(0, len(blob), REC_SIZE):
            raw = blob[off:off + REC_SIZE]
            if raw[0] == 3:          # 终止/跳过记录（LoadTBL 对 name[0]=#03 Continue）
                continue
            name = raw[0:4].decode('ascii', 'replace').rstrip('\x00 ')
            tbl.records.append(TBLRecord(raw=raw, name=name, identifier=raw[11]))
        # LoadTBL 后处理：FILE 记录缺失（或对应文件不存在，CLI 不复刻文件检查）→ = name
        if not tbl.tsfilename:
            tbl.set_str_field('FILE', tbl.file_stem)
        return tbl

    def save(self, fname: str):
        with open(fname, 'wb') as f:
            for r in self.records:
                f.write(r.raw)
            f.write(TERMINATOR)

    def save_phoenix(self, out_dir: str, site_name: str, ts_writer) -> str:
        """等价 TSiteTimeSeriesHelper.SavePhoenix：改 FILE/SITE 记录 + 落 TBL + 各 TS。"""
        out_dir = out_dir.replace('\\', '/')
        if not out_dir.endswith('/'):
            out_dir += '/'
        self.set_str_field('FILE', site_name)
        self.set_str_field('SITE', site_name)
        tbl_path = f'{out_dir}{site_name}.TBL'
        self.save(tbl_path)
        for ts in ts_writer:
            ts.write_phoenix(out_dir, self)
        return tbl_path
