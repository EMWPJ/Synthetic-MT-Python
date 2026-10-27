# -*- coding: utf-8 -*-
"""SEG 1.0 EDI 入料（非标准合成流程——必须向用户弹出警告）。

解析 MTDataPro/MTDP 输出的 SEG 1.0 EDI（>FREQ / >ZXXR..ZYYI / >TZXR..），
以实测阻抗 Z(f) 与倾子 T(f) 为该站响应，按 X/Y 单位极化基构造 XMLSITE
场指纹（H1=(1,0)/H2=(0,1)，E=Z·H，Hz=T·H）。

**警告合同（AGENTS.md 约定）**：EDI 携带实测噪声、局部畸变与编辑痕迹，
由它合成的"理论"只是该站实测响应的复刻——用它检验针对该站的处理
算法是循环论证，不能当作无偏真值基准。合法用途：按实测统计形态生成
可复现试验数据、受控噪声注入的对照组。CLI -edi 分支必须打印警告横幅
并在 JSON summary 里带 warning 字段。

单位：SEG 缺省 [mV/km]/[nT]；to_si=True 换算 SI ×1e-6/(1e-9/μ0)（与
synthts.modem 的阻抗换算同式）。EMPTY(1e32) 剔除为 NaN。
自包含实现（master 无 snrinv 包）；研究分支的 snrinv/edi.py 独立同源。
"""
import math
import re

import numpy as np

from .model import (Field, ForwardSite, densify_frequencies,
                    resample_site_log)

MU0 = 4e-7 * math.pi

_BLK = re.compile(r'^>\s*([A-Z0-9._]+)', re.M)


def _parse_blocks(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    head = {}
    for key in ('LAT', 'LONG', 'ELEV', 'DATAID', 'EMPTY'):
        m = re.search(r'^\s*%s\s*=\s*(.+)$' % key, txt, re.M)
        if m:
            head[key] = m.group(1).strip().strip('"')
    blocks = {}
    for m in _BLK.finditer(txt):
        name = m.group(1).split()[0].upper()
        start = txt.find('\n', m.end()) + 1     # 跳过 ">XXX // N" 标签行
        nxt = _BLK.search(txt, start)
        body = txt[start:nxt.start() if nxt else len(txt)]
        blocks.setdefault(name, []).append(body)
    return head, blocks


def _nums(body):
    return [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?',
                                         body.replace('D', 'E').replace('d', 'e'))
            if abs(float(v)) < 1e31]


def _dms(s):
    parts = re.findall(r'[-+]?\d+(?:\.\d+)?', s)
    v = 0.0
    for i, p in enumerate(parts[:3]):
        v += float(p) / (60.0 ** i)
    return v


def load_edi(path, to_si=True):
    """解析 EDI → dict(freqs, Z(2,2,n), T(2,n)或None, lat, lon, dataid)。

    Z 缺块/EMPTY(1e32) 置 NaN；to_si=True 换算 (V/m)/(A/m)。
    """
    head, blocks = _parse_blocks(path)
    if 'FREQ' not in blocks or not blocks['FREQ']:
        raise ValueError('EDI 无 >FREQ 频率块: %s' % path)
    freqs = np.asarray(_nums(blocks['FREQ'][0]), dtype=float)
    n = len(freqs)
    out = {'freqs': freqs, 'dataid': head.get('DATAID', ''),
           'lat': _dms(head['LAT']) if 'LAT' in head else None,
           'lon': _dms(head['LONG']) if 'LONG' in head else None}
    Z = np.full((2, 2, n), np.nan, dtype=complex)
    for rix, row in enumerate(('XX', 'XY', 'YX', 'YY')):
        i, j = divmod(rix, 2)
        vr = _nums(blocks['Z%sR' % row][0]) if 'Z%sR' % row in blocks else []
        vi = _nums(blocks['Z%sI' % row][0]) if 'Z%sI' % row in blocks else []
        for k in range(min(n, len(vr), len(vi))):
            Z[i, j, k] = complex(vr[k], vi[k])
    if to_si:
        Z = Z * (1e-6 / (1e-9 / MU0))
    T = None
    if 'TZXR' in blocks:
        T = np.full((2, n), np.nan, dtype=complex)
        for j, nm in enumerate(('TZX', 'TZY')):
            vr = _nums(blocks.get('%sR' % nm, [''])[0])
            vi = _nums(blocks.get('%sI' % nm, [''])[0])
            for k in range(min(n, len(vr), len(vi))):
                T[j, k] = complex(vr[k], vi[k])
    out['Z'] = Z
    out['T'] = T
    return out


def site_from_edi(edi, name=None, densify=2.0):
    """EDI dict → ForwardSite（单位极化基场指纹；NaN 频点剔除）。

    densify>0 时按每倍频程 densify 个点 log 域加密（锚点保留）。
    倾子缺失或该频点非有限时 Hz 置 0（不参与约束）。
    """
    freqs = edi['freqs']
    Z, T = edi['Z'], edi['T']
    m = np.isfinite(Z[0, 1]) & np.isfinite(Z[1, 0])
    fs = freqs[m]
    site = ForwardSite()
    site.name = name or edi['dataid'] or 'edi_site'
    site.x = edi['lon'] if edi['lon'] is not None else 0.0
    site.y = edi['lat'] if edi['lat'] is not None else 0.0
    for k in range(m.sum()):
        fld = Field(frequency=float(fs[k]), is_noise=False)
        Zk = Z[:, :, m][:, :, k]
        fld.ex1, fld.ey1 = Zk[0, 0], Zk[1, 0]     # Z @ (1,0)
        fld.ex2, fld.ey2 = Zk[0, 1], Zk[1, 1]     # Z @ (0,1)
        fld.hx1, fld.hy1 = 1.0 + 0j, 0.0 + 0j
        fld.hx2, fld.hy2 = 0.0 + 0j, 1.0 + 0j
        if T is not None:
            Tk = T[:, m][:, k]
            if np.isfinite(Tk).all():
                fld.hz1, fld.hz2 = Tk[0], Tk[1]
        site.fields.append(fld)
    if len(site.fields) < 2:
        raise ValueError('EDI 有效频点不足 2 个（离对角 Z 缺失/EMPTY 过多）'
                         ': %s' % (name or edi['dataid']))
    if densify and len(site.fields) > 1:
        site = resample_site_log(site, densify_frequencies(
            site.get_frequencies(False), densify))
    return site


EDI_WARNING = (
    '****************************************************************\n'
    '* 警告：EDI 实测阻抗入料是非标准合成流程\n'
    '*  - EDI 含实测噪声/局部畸变/编辑痕迹，合成"理论"=该站实测响应的复刻\n'
    '*  - 不能作为真值基准检验针对该站的处理算法（循环论证）\n'
    '*  - 合法用途：按实测统计形态生成可复现试验数据 / 受控噪声对照\n'
    '****************************************************************')
