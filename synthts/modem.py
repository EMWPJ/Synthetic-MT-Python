# -*- coding: utf-8 -*-
"""ModEM 格式兼容：正演响应/场值 → 合成测点（XMLSITE）。

ModEM 数据/响应文件为逐行 ASCII，每个记录一行：

    Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error
    1.000000E-03  001  0.0 0.0  -21000.0 19000.0 0.0  ZXY  1.66E+02 1.93E+02 2.0E+15

一个文件可含多个数据段（各段自带 `>` 头行：格式、单位、时间约定、
周期数/站数），逐段依次排列，例如改版 ModEM（EMWPJ）的正演文件：

    > Full_Impedance              ZXX/ZXY/ZYX/ZYY（复数）
    > Full_Vertical_Components    TX/TY 倾子（复数；原版名 TZX/TZY）
    > EM_Fields                   EPolEx..BPolHz —— 单位 X/Y 极化入射场
                                  在各站的响应，输入文件置 0、正演后回填，
                                  恰为 XMLSITE 的 10 复数场指纹

`> RhoPhase` 段的阻抗分量为 ρa[Ωm]/φ[°]，按 |Z|=sqrt(ωμ0·ρa) 重建。

测点构造（与 snrinv/edi.site_from_edi、tools/jianghan_modem_fp 同构）：
  有非零场值分量 → 直接取 10 复数（fields 路径）；
  否则 → H1=(1,0)/H2=(0,1) 单位极化基，E=Z·H，Hz=T·H（impedance 路径）。

约定对齐（AGENTS.md 电道合同）：
  - 相位口径实测（COMMEMI3D-2A 正演例）：ModEM 输出的 ZYX 相位
    −131.4°≡+48.6°，与场值段完全一致——即 +45° 型显示口径，与
    XMLSITE（layered_impedance，e^{−iωt}）和 EDI 相同，默认直通不
    共轭；仅当段头明写 exp(-i 时才共轭（conjugate='auto'）。
  - 单位默认换算到 SI：阻抗 [mV/km]/[nT]→(V/m)/(A/m) ×1e-6/(1e-9/μ0)，
    场值 E mV/km→V/m、H nT→A/m（B=μ0H）——与 snrinv/edi.to_si 同式；
    raw 保留文件原单位（合成不受影响：渲染按自然场幅值重归一 H）。
    注：实测 EM_Fields 段的 E 值与阻抗段差 ×1000（E 疑为 V/km）——
    差一个常数标度，被渲染端的自然幅值归一吸收，不影响合成与分离。

站位键：ModEM 以 (Code, X, Y) 唯一定站（COMMEMI 网格例中同一码位
可复用于多坐标）——同码多坐标时测点名加序号后缀。
"""
import math
import os

MU0 = 4e-7 * math.pi

_Z_COMPS = ('ZXX', 'ZXY', 'ZYX', 'ZYY')
_T_COMPS = ('TZX', 'TZY', 'TX', 'TY')
_F_COMPS = ('EPOLEX', 'EPOLEY', 'EPOLHX', 'EPOLHY', 'EPOLHZ',
            'BPOLEX', 'BPOLEY', 'BPOLHX', 'BPOLHY', 'BPOLHZ')

# 段格式 → 阻抗分量的取值解释（rhophase=ρa/φ，其余为复数 re/im）
_RHOPHASE_KEYS = ('rhophase', 'imp_phase')


class ModemResponse:
    """parse_modem_response 的结果（原始复数值，未共轭/未换算）。"""

    def __init__(self):
        self.sections = []           # [(name, units, time_sign), ...] 出现序
        self.fmt_z = ''              # 阻抗段的格式名（rhophase 判定用）
        self.periods = []            # 去重排序的周期(s)
        self.stations = {}           # code -> [(x, y), ...]（按出现序）
        self.coords = {}             # (code, x, y) -> (lat, lon)
        self.data = {}               # (code, x, y) -> {(period, comp): complex}
        self.fields_present = False  # 是否含 EPol*/BPol* 分量
        self.fields_nonzero = False  # 场值分量是否有非零（正演已回填）

    @property
    def rhophase(self):
        return any(k in self.fmt_z.lower().replace(' ', '_')
                   for k in _RHOPHASE_KEYS)

    @property
    def conversion_path(self):
        if self.fields_present and self.fields_nonzero:
            return 'fields'
        return 'impedance'

    def station_keys(self):
        """[(code, x, y), ...] 按站位出现序。"""
        out = []
        for code, xys in self.stations.items():
            for xy in xys:
                out.append((code, xy[0], xy[1]))
        return out


def _check_read(path, what='modem'):
    if '..' in str(path).replace(chr(92), '/'):
        raise ValueError('%s 路径不得包含 ..: %r' % (what, path))
    return path


def parse_modem_response(path):
    """读 ModEM 数据/响应文件 → ModemResponse（容忍残缺行，静默跳过）。

    段头为连续若干 `>` 行：格式名行起一段，随后的 exp/单位行并入当前段
    元数据，遇首条数据行（或下一段名/文件尾）时该段定稿。
    """
    _check_read(path)
    resp = ModemResponse()
    cur = None                        # {'name','units','sign'} 未定稿段

    def _flush():
        if cur and cur['name']:
            resp.sections.append((cur['name'], cur['units'], cur['sign']))
            if any(c in cur['name'] for c in ('impedance', 'rhophase')):
                resp.fmt_z = cur['name']

    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('>'):
                low = s.lower()
                name = ''
                for key in ('full_impedance', 'rhophase', 'imp_tipper',
                            'impedance_tipper', 'full_vertical_components',
                            'em_fields', 'impedance'):
                    if key in low.replace(' ', '_'):
                        name = key
                        break
                if name:
                    _flush()
                    cur = {'name': name, 'units': '', 'sign': ''}
                if 'exp' in low and 'i' in low.replace(' ', ''):
                    if cur is not None:
                        cur['sign'] = ('-i' if '-i' in low.replace(' ', '')
                                       else '+i')
                if '[' in s and cur is not None:
                    cur['units'] = s.lstrip('> ')
                continue
            if cur is not None:
                _flush()
                cur = None
            t = s.split()
            if len(t) < 10:
                continue
            try:
                per = float(t[0])
                code = t[1]
                lat = float(t[2])
                lon = float(t[3])
                x = float(t[4])
                y = float(t[5])
                comp = t[7].upper()
                re_, im_ = float(t[8]), float(t[9])
            except (ValueError, IndexError):
                continue
            if per not in resp.periods:
                resp.periods.append(per)
            xys = resp.stations.setdefault(code, [])
            if (x, y) not in xys:
                xys.append((x, y))
            resp.coords[(code, x, y)] = (lat, lon)
            resp.data.setdefault((code, x, y), {})[(per, comp)] = \
                complex(re_, im_)
            if comp in _F_COMPS:
                resp.fields_present = True
                if re_ != 0.0 or im_ != 0.0:
                    resp.fields_nonzero = True
    resp.periods.sort()
    _flush()
    return resp


def _impedance_rows(resp, key, to_si):
    """一个站位 → [(freq, Z(2,2), T(2)或None)]（SI 或原单位；未共轭）。"""
    rows = resp.data.get(key, {})
    by_per = {}
    for (per, comp), v in rows.items():
        by_per.setdefault(per, {})[comp] = v
    out = []
    for per in sorted(by_per):
        d = by_per[per]
        if not all(c in d for c in _Z_COMPS):
            continue
        if resp.rhophase:
            # ρa[Ωm]/φ[°] 重建的 Z 已是 SI（V/m per A/m），不再换算
            w = 2.0 * math.pi / per
            amp = [math.sqrt(w * MU0 * abs(d[c].real))
                   * complex(math.cos(math.radians(d[c].imag)),
                             math.sin(math.radians(d[c].imag)))
                   for c in _Z_COMPS]
        else:
            amp = [d[c] for c in _Z_COMPS]
            if to_si:
                amp = [a * (1e-6 / (1e-9 / MU0)) for a in amp]
        Z = [[amp[0], amp[1]], [amp[2], amp[3]]]
        T = None
        tc = {}
        for (p2, comp), v in rows.items():
            if p2 == per and comp in _T_COMPS:
                tc[comp] = v
        if len(tc) >= 2:
            tzx = tc.get('TZX', tc.get('TX'))
            tzy = tc.get('TZY', tc.get('TY'))
            if tzx is not None and tzy is not None:
                T = (tzx, tzy)
        out.append((1.0 / per, Z, T))
    return out


def sites_from_response(resp, to_si=True, conjugate='auto'):
    """ModemResponse → [ForwardSite]（XMLSITE e^{−iωt} 约定的合成测点）。

    conjugate: 'auto'（仅段头明写 exp(-i 时共轭；实测 ModEM 输出与
               XMLSITE/EDI 同为 +45° 口径，默认直通）/ True / False。
               共轭 = 相位约定变换（e^{±iωt} 互逆）：所有场相位取反。
               to_si=False 保留文件原单位。
    至少 2 个有效频点才成站（FingerprintMap 下限）。
    """
    from .model import Field, ForwardSite
    if conjugate == 'auto':
        conj = (len(resp.sections) > 0
                and all(sign == '-i' for _, _, sign in resp.sections))
    else:
        conj = bool(conjugate)

    def cv(z):
        return complex(z.real, -z.imag) if conj else z

    e_fac = 1e-6 if to_si else 1.0                    # mV/km -> V/m
    h_fac = (1e-9 / MU0) if to_si else 1.0            # nT -> A/m

    sites = []
    for key in resp.station_keys():
        code, x, y = key
        lat, lon = resp.coords.get(key, (0.0, 0.0))
        name = code if len(resp.stations.get(code, [])) == 1 \
            else '%s_%d' % (code,
                            resp.stations[code].index((x, y)) + 1)
        site = ForwardSite()
        site.name = name
        site.x, site.y = lon, lat      # 与 jh_fp_modem 同口径：X=经度 Y=纬度
        built = 0
        if resp.conversion_path == 'fields':
            by_per = {}
            for (per, comp), v in resp.data.get(key, {}).items():
                by_per.setdefault(per, {})[comp] = v
            for per in sorted(by_per):
                d = by_per[per]
                if not all(c in d for c in _F_COMPS):
                    continue
                fv = {c: cv(d[c]) for c in _F_COMPS}
                fld = Field(frequency=1.0 / per, is_noise=False)
                fld.ex1, fld.ey1 = fv['EPOLEX'] * e_fac, fv['EPOLEY'] * e_fac
                fld.hx1, fld.hy1 = fv['EPOLHX'] * h_fac, fv['EPOLHY'] * h_fac
                fld.hz1 = fv['EPOLHZ'] * h_fac
                fld.ex2, fld.ey2 = fv['BPOLEX'] * e_fac, fv['BPOLEY'] * e_fac
                fld.hx2, fld.hy2 = fv['BPOLHX'] * h_fac, fv['BPOLHY'] * h_fac
                fld.hz2 = fv['BPOLHZ'] * h_fac
                site.fields.append(fld)
                built += 1
        else:
            for freq, Z, T in _impedance_rows(resp, key, to_si):
                h1 = complex(1.0, 0.0), complex(0.0, 0.0)
                h2 = complex(0.0, 0.0), complex(1.0, 0.0)
                ex1 = cv(Z[0][0] * h1[0] + Z[0][1] * h1[1])
                ey1 = cv(Z[1][0] * h1[0] + Z[1][1] * h1[1])
                ex2 = cv(Z[0][0] * h2[0] + Z[0][1] * h2[1])
                ey2 = cv(Z[1][0] * h2[0] + Z[1][1] * h2[1])
                fld = Field(frequency=freq, is_noise=False)
                fld.ex1, fld.ey1 = ex1, ey1
                fld.ex2, fld.ey2 = ex2, ey2
                fld.hx1, fld.hy1 = h1
                fld.hx2, fld.hy2 = h2
                if T is not None:
                    fld.hz1 = cv(T[0])
                    fld.hz2 = cv(T[1])
                site.fields.append(fld)
                built += 1
        if built >= 2:
            sites.append(site)
    return sites


def write_modem_sites(sites, out_dir):
    """合成测点落盘 <name>.XMLSITE → 路径列表（拒 .. 与路径分隔符）。"""
    from .model import save_site
    out_dir = str(out_dir)
    if '..' in out_dir.replace(chr(92), '/'):
        raise ValueError('输出目录不得包含 ..: %r' % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    seen = set()
    for site in sites:
        name = site.name or 'site'
        if any(ch in name for ch in '/\\:'):
            raise ValueError('测点名含非法字符: %r' % name)
        if name in seen:
            raise ValueError('测点名重复（输出会互相覆盖）: %s' % name)
        seen.add(name)
        path = os.path.join(out_dir, name + '.XMLSITE')
        save_site(site, path)
        paths.append(path)
    return paths
