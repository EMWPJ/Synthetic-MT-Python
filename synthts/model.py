# -*- coding: utf-8 -*-
"""XMLSITE 正演场模型读取（ForwardSiteXML schema 等价移植）。

格式（与 MTDataPro 合成界面互认）:
  <Site name="SYN001" X="100.5" Y="30.2">
    <Synthetic>False</Synthetic>
    <Fields frequency="320" isNoise="False">   ← frequency/isNoise 是属性!
      <Zxx>re,im</Zxx> ... <Hz2>re,im</Hz2>     ← 复数对是子元素文本 "re,im"/"re im"
    </Fields>...
  </Site>
"""
import xml.etree.ElementTree as ET
import math
from dataclasses import dataclass


@dataclass
class Field:
    frequency: float = 0.0
    is_noise: bool = False
    zxx: complex = 0j
    zxy: complex = 0j
    zyx: complex = 0j
    zyy: complex = 0j
    tzx: complex = 0j
    tzy: complex = 0j
    ex1: complex = 0j
    ey1: complex = 0j
    hx1: complex = 0j
    hy1: complex = 0j
    hz1: complex = 0j
    ex2: complex = 0j
    ey2: complex = 0j
    hx2: complex = 0j
    hy2: complex = 0j
    hz2: complex = 0j


@dataclass
class ForwardSite:
    name: str = ''
    x: float = 0.0
    y: float = 0.0
    synthetic: bool = False
    fields: list = None

    def __post_init__(self):
        if self.fields is None:
            self.fields = []

    # —— TForwardSiteHelper 等价（SynthForwardSiteHelper.pas）——
    def get_frequencies(self, noise: bool) -> list:
        return [f.frequency for f in self.fields if f.is_noise == noise]

    def get_all_frequencies(self) -> list:
        return [f.frequency for f in self.fields]

    def get_feh(self, noise: bool = False):
        """返回 (fre, Ex1..Hz2 共 10 组) —— GetFEH 同过滤语义。"""
        rows = [f for f in self.fields if f.is_noise == noise]
        fre = [f.frequency for f in rows]
        cols = [[getattr(f, n) for f in rows] for n in
                ('ex1', 'ey1', 'hx1', 'hy1', 'hz1', 'ex2', 'ey2', 'hx2', 'hy2', 'hz2')]
        return [fre] + cols

    def add_calibration(self, responds):
        """AddCalibration: 场幅值除 responds（responds=[5 通道][频点] 复数）+ 磁道 4e-7π。

        E_adc = E/rE、H_adc = H*mu0/rH（mu0 = 4e-7*pi，磁场 A/m -> B 的
        物理换算），处理端乘回即可还原。"""
        if len(responds) < 5 or len(responds[0]) < 1:
            return
        cnt = min(len(self.fields), len(responds[0]))
        mu0 = 4e-7 * 3.141592653589793
        for i in range(cnt):
            f = self.fields[i]
            f.ex1 /= responds[0][i]
            f.ex2 /= responds[0][i]
            f.ey1 /= responds[1][i]
            f.ey2 /= responds[1][i]
            f.hx1 = f.hx1 / responds[2][i] * mu0
            f.hx2 = f.hx2 / responds[2][i] * mu0
            f.hy1 = f.hy1 / responds[3][i] * mu0
            f.hy2 = f.hy2 / responds[3][i] * mu0
            f.hz1 = f.hz1 / responds[4][i] * mu0
            f.hz2 = f.hz2 / responds[4][i] * mu0

    def update_nature_magnetic_amplitude(self, escale, hscale):
        index = 0
        for f in self.fields:
            if f.is_noise:
                continue
            f.ex1 *= escale[index]
            f.ex2 *= hscale[index]
            f.ey1 *= escale[index]
            f.ey2 *= hscale[index]
            f.hx1 *= escale[index]
            f.hx2 *= hscale[index]
            f.hy1 *= escale[index]
            f.hy2 *= hscale[index]
            f.hz1 *= escale[index]
            f.hz2 *= hscale[index]
            index += 1


def _parse_complex(text: str) -> complex:
    """TComplex.CreateFromXML: "re,im" / "re im"。"""
    s = text.strip()
    for sep in (',', ' '):
        if sep in s:
            parts = s.replace(',', ' ').split()
            return complex(float(parts[0]), float(parts[1]))
    return complex(float(s), 0.0)


def _str2bool(text: str) -> bool:
    t = text.strip().lower()
    if t in ('1', 'true', 't', 'yes'):
        return True
    if t in ('0', 'false', 'f', 'no', ''):
        return False
    return bool(int(float(t)))


_COMPLEX_NAMES = ('Zxx', 'Zxy', 'Zyx', 'Zyy', 'Tzx', 'Tzy',
                  'Ex1', 'Ey1', 'Hx1', 'Hy1', 'Hz1',
                  'Ex2', 'Ey2', 'Hx2', 'Hy2', 'Hz2')


def load_site(fname: str) -> ForwardSite:
    tree = ET.parse(fname)
    root = tree.getroot()
    site = ForwardSite()
    site.name = root.get('name', '')
    site.x = float(root.get('X', '0') or 0)
    site.y = float(root.get('Y', '0') or 0)
    for node in root:
        tag = node.tag.lower() if isinstance(node.tag, str) else ''
        if tag == 'synthetic':
            site.synthetic = _str2bool(node.text or 'False')
        elif tag == 'fields':
            f = Field()
            freq_attr = node.get('frequency')
            if freq_attr is not None:
                f.frequency = float(freq_attr)
            noise_attr = node.get('isNoise')
            if noise_attr is not None:
                f.is_noise = _str2bool(noise_attr)
            for sub in node:
                if isinstance(sub.tag, str) and sub.tag in _COMPLEX_NAMES:
                    setattr(f, sub.tag.lower(), _parse_complex(sub.text or '0,0'))
            site.fields.append(f)
    return site


# ── 对数域频点加密（Wang et al. 2023 频点集的自动加密扩展）───────────────

def densify_frequencies(freqs, per_octave=4.0):
    """对数域频点加密：相邻锚点间按每倍频程 per_octave 个点均匀(log 域)插入。

    原频点作为锚点精确保留（插值点不移动锚点）；per_octave<=0 时原样返回。
    Wang et al. (2023) 使用的频点约 2 个/倍频程（MTU-A 标准表），稀疏频点使
    合成时序谱仅在目标频点有能量（其 PSD 观测：目标频点比非目标高一个量级）。
    加密后谱连续性接近天然场，且窗口 DFT 对全部满足 E=ZH 的分量是线性 Z
    保持的——跨频泄漏不损害阻抗估计（见 mt_e2e_demo 全网格验证）。
    """
    freqs = sorted(set(float(f) for f in freqs))
    if per_octave <= 0 or len(freqs) < 2:
        return list(freqs)
    out = [freqs[0]]
    for a, b in zip(freqs[:-1], freqs[1:]):
        if b <= a:
            continue
        n_oct = math.log2(b / a)
        n = max(1, int(math.ceil(n_oct * per_octave)))
        for k in range(1, n):
            out.append(a * (b / a) ** (k / n))
        out.append(b)
    return out


_FIELD_COMPONENTS = ('ex1', 'ey1', 'hx1', 'hy1', 'hz1',
                     'ex2', 'ey2', 'hx2', 'hy2', 'hz2')


def resample_site_log(site, new_freqs):
    """把站点信号频点的 10 个场复分量在 log10(f) 域线性插值到新频表。

    锚点(原频点)处插值严格等于原值；返回新 ForwardSite（仅信号频点，
    噪声行不随迁——合成只用信号行）。测点名/坐标保留。
    """
    import math as _m
    src_freqs = sorted(site.get_frequencies(False))
    if not src_freqs:
        raise ValueError('模型无信号频点')
    rows = {f: None for f in src_freqs}
    for f in site.fields:
        if not f.is_noise and f.frequency in rows:
            rows[f.frequency] = f
    lf = [_m.log10(f) for f in src_freqs]

    def interp_comp(comp, x):
        """单复分量在 log10 频域的线性插值（re/im 分别）。"""
        re = [getattr(rows[f], comp).real for f in src_freqs]
        im = [getattr(rows[f], comp).imag for f in src_freqs]
        return complex(_interp1(lf, re, x), _interp1(lf, im, x))

    new = ForwardSite()
    new.name = site.name
    new.x = site.x
    new.y = site.y
    new.synthetic = site.synthetic
    for f in sorted(new_freqs):
        x = _m.log10(f)
        if f in rows:
            import copy as _copy
            new.fields.append(_copy.deepcopy(rows[f]))
            new.fields[-1].frequency = f
            continue
        fld = Field()
        fld.frequency = f
        fld.is_noise = False
        for comp in _FIELD_COMPONENTS:
            setattr(fld, comp, interp_comp(comp, x))
        new.fields.append(fld)
    return new


def _interp1(xs, ys, x):
    """单调 xs 上的线性插值（端点钳制）。"""
    import bisect as _bs
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = _bs.bisect_right(xs, x) - 1
    if i >= len(xs) - 1:
        return ys[-1]
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] * (1.0 - t) + ys[i + 1] * t


def save_site(site, path):
    """写回 XMLSITE（与 load_site 同 schema；frequency/isNoise 是属性）。"""
    def cstr(z):
        return f"{float(z.real)!r},{float(z.imag)!r}"
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<Site name="%s" X="%r" Y="%r">' % (site.name, site.x, site.y),
             '  <Synthetic>%s</Synthetic>' % ('True' if site.synthetic else 'False')]
    for f in site.fields:
        lines.append('  <Fields frequency="%r" isNoise="%s">'
                     % (f.frequency, 'True' if f.is_noise else 'False'))
        lines.append('    <Zxx>0,0</Zxx><Zxy>0,0</Zxy><Zyx>0,0</Zyx><Zyy>0,0</Zyy>'
                     '<Tzx>0,0</Tzx><Tzy>0,0</Tzy>')
        for tag in ('Ex1', 'Ey1', 'Hx1', 'Hy1', 'Hz1',
                    'Ex2', 'Ey2', 'Hx2', 'Hy2', 'Hz2'):
            lines.append('    <%s>%s</%s>' % (tag, cstr(getattr(f, tag.lower())), tag))
        lines.append('  </Fields>')
    lines.append('</Site>')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(chr(10).join(lines) + chr(10))
