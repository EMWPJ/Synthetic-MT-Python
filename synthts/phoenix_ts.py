# -*- coding: utf-8 -*-
"""Phoenix TS2-5 二进制写盘（向量化）。

记录布局: 32 字节头 + Channels*Scans*3 字节样本（scan 主序/通道交错，
每样本 24bit 小端 [+0]=low [+1]=mid [+2]=high）。
头（自然对齐逐字段，全部小端）:
  [0]秒 [1]分 [2]时 [3]日 [4]月 [5]年%100 [6]星期(1=Mon..7=Sun) [7]世纪
  [8:10]BoxSN(i16) [10:12]Scans(i16) [12]Channels [13]TSLength=32
  [14]Status [15]Saturation [16]Reserved [17]SampleLength=3
  [18:20]SampleRate(i16) [20]SRUnits [21]ClockStatus=4 [22:26]ClockError(i32) [26:32]=0
样本量化: 分档记录 np.rint（银行家舍入）、连续档 np.trunc；
超 ±2^23 截断回卷（and 0xFFFFFF 语义）。
"""
import numpy as np

from .tbl import TBL

CH_COUNT = 5

# TS 文件参数表（scans=每记录扫描数, samplerate=记录头采样率）
_TS_SPEC = {
    'TS2': dict(scans=2400, samplerate=24000),
    'TS3': dict(scans=2400, samplerate=2400),
    'TS4': dict(scans=150, samplerate=150),
    'TS5': dict(scans=15, samplerate=15),
}


def _civil_from_days(days):
    """1970-01-01 起的天数(int64 数组) -> (y, m, d)（Hinnant 公历算法向量化）。"""
    z = days + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = np.where(mp < 10, mp + 3, mp - 9)
    y = y + (m <= 2).astype(np.int64)
    return y, m, d


def _headers_block(base_tdatetime, nrec, scans, ts_sr, boxsn, samplerate):
    """一个时段全部记录的 32 字节头（向量化）。"""
    offs = np.arange(nrec) * scans / (ts_sr * 3600.0 * 24.0)
    isec = np.round((base_tdatetime + offs) * 86400.0).astype(np.int64)
    days = np.floor_divide(isec, 86400)             # TDateTime 天数(1899-12-30 纪元)
    sod = isec - days * 86400
    y, mo, d = _civil_from_days(days - 25569)       # -> 1970 纪元(1970-01-01=43831-18262)
    head = np.zeros((nrec, 32), np.uint8)
    head[:, 0] = sod % 60
    head[:, 1] = (sod // 60) % 60
    head[:, 2] = sod // 3600
    head[:, 3] = d
    head[:, 4] = mo
    head[:, 5] = y % 100
    d1970 = days - 25569
    head[:, 6] = ((d1970 + 3) % 7) + 1        # 1970-01-01 周四 -> 1=Mon..7=Sun
    head[:, 7] = y // 100
    head[:, 8] = boxsn & 0xFF
    head[:, 9] = (boxsn >> 8) & 0xFF
    head[:, 10] = scans & 0xFF
    head[:, 11] = (scans >> 8) & 0xFF
    head[:, 12] = CH_COUNT
    head[:, 13] = 32
    head[:, 17] = 3
    head[:, 18] = samplerate & 0xFF
    head[:, 19] = (samplerate >> 8) & 0xFF
    head[:, 21] = 4                             # ClockStatus
    return head


def _pack_body(X5, continuous):
    """(5, n) float -> n*15 字节样本体（scan 主序/通道交错，24bit LE）。

    电道 (Ex, Ey) 写盘取负：正演场 E=ZH 为 e^{-iωt} 阻抗口径，处理链按
    e^{+iωt} 估计阻抗，两侧电道对称取负后相位才与 MT 惯例一致。
    """
    X5 = X5.copy()
    X5[0:2] = -X5[0:2]
    v = np.trunc(X5) if continuous else np.rint(X5)
    v = v.astype(np.int64)
    np.bitwise_and(v, 0xFFFFFF, out=v)
    v = v.T.reshape(-1)
    body = np.empty((v.size, 3), np.uint8)
    body[:, 0] = v & 0xFF
    body[:, 1] = (v >> 8) & 0xFF
    body[:, 2] = (v >> 16) & 0xFF
    return body.reshape(-1)


def write_time_series(ts, out_dir: str, tbl: TBL, site_name: str):
    """写一个 TS 档: Continuous(TS4/TS5, trunc) 与 Multiple(TS2/TS3/TS4, rint)。"""
    spec = _TS_SPEC[ts['name']]
    if '..' in out_dir.replace(chr(92), '/') or '..' in site_name:
        raise ValueError('output path must not traverse up')
    boxsn = int(tbl.snum) if str(tbl.snum).strip().lstrip('-').isdigit() else 0
    out = f'{out_dir}{site_name}.{ts["name"]}'
    scans = spec['scans']
    body_len = scans * CH_COUNT * 3
    with open(out, 'wb') as f:
        for seg_idx, begin in enumerate(ts['begins']):
            X = np.asarray([ts['channels'][c][seg_idx] for c in range(5)])
            nrec = X.shape[1] // scans
            if nrec == 0:
                continue
            if nrec * scans != X.shape[1]:
                X = X[:, :nrec * scans]
            head = _headers_block(begin, nrec, scans, ts['sample_rate'],
                                  boxsn, spec['samplerate'])
            body = _pack_body(X, ts['continuous']).reshape(nrec, body_len)
            rec = np.empty((nrec, 32 + body_len), np.uint8)
            rec[:, :32] = head
            rec[:, 32:] = body
            del head, body, X
            f.write(rec.tobytes())
            del rec
    return out
