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

`> RhoPhase` 段（真实拼写 Rho_Phase / Off_Diagonal_Rho_Phase，匹配时
归一化空格与下划线）的阻抗为 RHOXX/PHSXX..RHOYY/PHSYX 独立分量行
（每行单值 + 误差列，ModEM 默认仅离对角 RHOXY/PHSXY/RHOYX/PHSYX），
按 |Z|=sqrt(ωμ0·ρa)·e^{iφ} 重建（已是 SI）；缺省对角置 0（离对角
标准输出的既定实践，与 jh_fp_modem 转换一致）。

测点构造（与 snrinv/edi.site_from_edi、tools/jianghan_modem_fp 同构）：
  有非零场值分量 → 直接取 10 复数（fields 路径）；
  否则 → H1=(1,0)/H2=(0,1) 单位极化基，E=Z·H，Hz=T·H（impedance 路径）。

约定对齐（AGENTS.md 电道合同）：
  - 相位口径实测（COMMEMI3D-2A 正演例）：ModEM 输出的 ZYX 相位
    −131.4°≡+48.6°，与场值段完全一致——即 +45° 型显示口径，与
    XMLSITE（layered_impedance，e^{−iωt}）和 EDI 相同，默认直通不
    共轭；仅当段头明写 exp(-i 时才共轭（conjugate='auto'）。
  - 单位默认换算到 SI：阻抗 [mV/km]/[nT]→(V/m)/(A/m) ×1e-6/(1e-9/μ0)，
    与 snrinv/edi.to_si 同式。**EM_Fields 段实为 SI：E [V/m]，H 是磁通
    密度 B [T]**（实测 COMMEMI 例子：由场值重建的 Z=E·H⁻¹ 与阻抗段
    SI 值在长周期一致到 1e-6~1e-7；段头 [mV/km]/[nT] 是照抄阻抗段的
    摆设）——换算为 E 直通、H×1/μ0 → A/m。raw 保留文件原单位。
    E/H 比值直接决定合成阻抗（ρa∝|Z|²），任何标度错误都不会被渲染端
    的自然幅值归一吸收——归一按极化列整体缩放。
    注：场值段与阻抗段是 exe 的两条求值路径，粗网格最短周期（趋肤深度
    ≪ 网格）可分叉 ~20%（COMMEMI 例 T=0.001 s 实测；长周期 1e-6 级）。
    场值路径以场值段为准；需与阻抗严格一致时用纯阻抗文件。

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

# 段名匹配按归一化（去空格/下划线）子串：接住 Full_Impedance / RhoPhase /
# Rho_Phase / Off_Diagonal_Rho_Phase / Imp_Tipper / Full_Vertical_Components
# / EM_Fields 等真实拼写（实测 ID003 集群响应为 Off_Diagonal_Rho_Phase）
_SECTION_TYPES = (('fullimpedance', 'full_impedance'),
                  ('rhophase', 'rhophase'),
                  ('imptipper', 'imp_tipper'),
                  ('impedancetipper', 'impedance_tipper'),
                  ('fullverticalcomponents', 'full_vertical_components'),
                  ('emfields', 'em_fields'),
                  ('impedance', 'impedance'))

# ρ/φ 独立分量行（单值 + 误差列）；对角元素缺省置 0（离对角标准输出）
_RHO_PAIRS = ('XX', 'XY', 'YX', 'YY')


class ModemResponse:
    """parse_modem_response 的结果（原始复数值，未共轭/未换算）。"""

    def __init__(self):
        self.sections = []           # [(name, units, time_sign), ...] 出现序
        self.fmt_z = ''              # 阻抗段的格式名（rhophase 判定用）
        self.periods = []            # 去重排序的周期(s)
        self.stations = {}           # code -> [(x, y), ...]（按出现序）
        self.coords = {}             # (code, x, y) -> (lat, lon, elev_m)
        self.record_order = []       # (per, code, x, y) 文件首现序（跨段去重）
        self.header_proto = {}       # 首段头行原型：exp/elev/origin（含 '>'）
        self.data = {}               # (code, x, y) -> {(period, comp): complex}
        self.raw_row = {}            # (per, code, x, y) -> 前 7 列原始 token
        self.comps_present = set()   # 出现过的分量名（诊断用）
        self.fields_present = False  # 是否含 EPol*/BPol* 分量
        self.fields_nonzero = False  # 场值分量是否有非零（正演已回填）

    @property
    def has_em_fields(self):
        return any(name == 'em_fields' for name, _, _ in self.sections)

    @property
    def rhophase(self):
        norm = self.fmt_z.lower().replace(' ', '').replace('_', '')
        return 'rhophase' in norm

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
    元数据，遇首条数据行（或下一段名/文件尾）时该段定稿。首段的 exp/
    单值行/坐标行另存 header_proto（add_em_fields 重建段头用）。
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
                norm = low.replace(' ', '').replace('_', '')
                name = ''
                for key, canon in _SECTION_TYPES:
                    if key in norm:
                        name = canon
                        break
                if name:
                    _flush()
                    cur = {'name': name, 'units': '', 'sign': ''}
                if 'exp' in low and 'i' in low.replace(' ', ''):
                    if cur is not None:
                        cur['sign'] = ('-i' if '-i' in low.replace(' ', '')
                                       else '+i')
                    if cur is not None and len(resp.sections) == 0 \
                            and 'exp' not in resp.header_proto:
                        resp.header_proto['exp'] = s
                if '[' in s and cur is not None:
                    cur['units'] = s.lstrip('> ')
                t = s.lstrip('> ').split()
                if len(t) == 1:
                    try:
                        float(t[0])
                        if len(resp.sections) == 0 and cur is not None:
                            resp.header_proto.setdefault('elev_line', s)
                    except ValueError:
                        pass
                elif len(t) == 2:
                    try:
                        float(t[0]), float(t[1])
                        # 纯整数对（无小数点/指数）是周期数/站数行——
                        # 不得误当坐标原点行复刻（会让补段头多出一行）
                        is_counts = all('.' not in v and 'e' not in v.lower()
                                        for v in t)
                        if (not is_counts and len(resp.sections) == 0
                                and cur is not None):
                            resp.header_proto.setdefault('origin_line', s)
                    except ValueError:
                        pass
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
                elev = float(t[6])
                comp = t[7].upper()
                if comp.startswith(('RHO', 'PHS')):
                    # ρ/φ 独立分量行：单值列 + 误差列（t[9] 不是虚部！）
                    re_, im_ = float(t[8]), 0.0
                else:
                    re_, im_ = float(t[8]), float(t[9])
            except (ValueError, IndexError):
                continue
            if per not in resp.periods:
                resp.periods.append(per)
            xys = resp.stations.setdefault(code, [])
            if (x, y) not in xys:
                xys.append((x, y))
            resp.coords[(code, x, y)] = (lat, lon, elev)
            if (per, code, x, y) not in resp.record_order:
                resp.record_order.append((per, code, x, y))
            resp.comps_present.add(comp)
            resp.data.setdefault((code, x, y), {})[(per, comp)] = \
                complex(re_, im_)
            resp.raw_row.setdefault((per, code, x, y), t[:7])
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
        if resp.rhophase:
            # RHO*/PHS* 独立分量行 → Z = sqrt(ωμ0·ρa)·e^{iφ}（已是 SI）。
            # 离对角齐备才有效；缺省对角置 0（ModEM 默认离对角输出的
            # 既定实践，与 jh_fp_modem 转换一致）
            if not all(('RHO' + ij) in d and ('PHS' + ij) in d
                       for ij in ('XY', 'YX')):
                continue
            w = 2.0 * math.pi / per

            def z_el(ij):
                rho_v = d.get('RHO' + ij)
                ph_v = d.get('PHS' + ij)
                if rho_v is None or ph_v is None or rho_v.real <= 0:
                    return 0j               # 缺省（离对角文件的对角）置 0
                return math.sqrt(w * MU0 * rho_v.real) * complex(
                    math.cos(math.radians(ph_v.real)),
                    math.sin(math.radians(ph_v.real)))

            amp = [z_el(ij) for ij in _RHO_PAIRS]
        else:
            if not all(c in d for c in _Z_COMPS):
                continue
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
    转换路径逐站判定：该站有非零场值分量 → fields 路径（全零占位行跳过，
    未回填/部分回填的站自动回退 impedance 路径）。至少 2 个有效频点才
    成站（FingerprintMap 下限）。
    """
    from .model import Field, ForwardSite
    if conjugate == 'auto':
        conj = (len(resp.sections) > 0
                and all(sign == '-i' for _, _, sign in resp.sections))
    else:
        conj = bool(conjugate)

    def cv(z):
        return complex(z.real, -z.imag) if conj else z

    # 阻抗换算在 _impedance_rows 内完成（[mV/km]/[nT]→SI）。
    # EM_Fields 实为 SI：E [V/m]、H 是 B [T] → H[A/m] ×1/μ0
    # （E/H 比值不受影响，恰为该站阻抗）。
    fe_fac = 1.0                                       # E 已是 V/m
    fh_fac = (1.0 / MU0) if to_si else 1.0             # B[T] -> H[A/m]

    sites = []
    for key in resp.station_keys():
        code, x, y = key
        lat, lon = resp.coords.get(key, (0.0, 0.0, 0.0))[:2]
        name = code if len(resp.stations.get(code, [])) == 1 \
            else '%s_%d' % (code,
                            resp.stations[code].index((x, y)) + 1)
        site = ForwardSite()
        site.name = name
        site.x, site.y = lon, lat      # 与 jh_fp_modem 同口径：X=经度 Y=纬度
        built = 0
        rows = resp.data.get(key, {})
        st_fields = any(v != 0 for (p, c), v in rows.items() if c in _F_COMPS)
        if st_fields:
            by_per = {}
            for (per, comp), v in rows.items():
                by_per.setdefault(per, {})[comp] = v
            for per in sorted(by_per):
                d = by_per[per]
                if not all(c in d for c in _F_COMPS):
                    continue
                if all(d[c] == 0 for c in _F_COMPS):
                    continue                     # 未回填的占位行
                fv = {c: cv(d[c]) for c in _F_COMPS}
                fld = Field(frequency=1.0 / per, is_noise=False)
                fld.ex1 = fv['EPOLEX'] * fe_fac
                fld.ey1 = fv['EPOLEY'] * fe_fac
                fld.hx1, fld.hy1 = fv['EPOLHX'] * fh_fac, fv['EPOLHY'] * fh_fac
                fld.hz1 = fv['EPOLHZ'] * fh_fac
                fld.ex2 = fv['BPOLEX'] * fe_fac
                fld.ey2 = fv['BPOLEY'] * fe_fac
                fld.hx2, fld.hy2 = fv['BPOLHX'] * fh_fac, fv['BPOLHY'] * fh_fac
                fld.hz2 = fv['BPOLHZ'] * fh_fac
                site.fields.append(fld)
                built += 1
        if built < 2:
            built = 0
            site = ForwardSite()
            site.name = name
            site.x, site.y = lon, lat
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


# ── EM_Fields 占位段补充（改版 ModEM 正演的前置准备）─────────────────────
#
# 常规 ModEM 的 dat 只有阻抗/倾子段；改版 ModEM（EMWPJ）约定：在数据文件
# 末尾追加一个 `> EM_Fields` 段，每 (周期, 测站) 一组 EPolEx..BPolHz 十行、
# 初值 0、误差列 2.0E+15（inversion 忽略量），正演后程序把单位 X/Y 极化
# 入射场的各站响应原位回填（COMMEMI3D-2A 例子即此结构）。本函数按源文件
# 自己的 (周期, 测站) 首现序生成该段——行枚举与源文件的阻抗段完全一致。

_EM_COMPS = ('EPolEx', 'EPolEy', 'EPolHx', 'EPolHy', 'EPolHz',
             'BPolEx', 'BPolEy', 'BPolHx', 'BPolHy', 'BPolHz')


def _safe_path(p, what):
    """显式路径守卫：拒 '..'（原始串与归一化结果都不得含 ..），返回规范化绝对路径。"""
    raw = str(p).replace(chr(92), '/')
    if '..' in raw.split('/'):
        raise ValueError('%s 路径不得包含 ..: %r' % (what, p))
    norm = os.path.normpath(os.path.abspath(str(p)))
    if '..' in norm.replace(chr(92), '/').split('/'):
        raise ValueError('%s 路径不得包含 ..: %r' % (what, p))
    return norm


def add_em_fields(src, dst):
    """常规 ModEM dat → 追加 EM_Fields 占位段的新 dat（源内容逐字节保留，
    仅当缺尾换行时补一个换行防注释并入数据行）。

    返回 dict(n_periods, n_stations, n_records, n_rows)。源文件已含
    EM_Fields 段、无数据行、或 src==dst 时拒绝。
    """
    src = _safe_path(src, 'modem 输入')
    dst = _safe_path(dst, 'modem 输出')
    if src == dst:
        raise ValueError('输出文件须与输入不同: %r' % src)
    resp = parse_modem_response(src)
    if resp.has_em_fields:
        raise ValueError('源文件已含 EM_Fields 段: %s' % src)
    if not resp.record_order:
        raise ValueError('ModEM 文件无数据行: %s' % src)
    proto = resp.header_proto
    exp_line = proto.get('exp', '>  exp(i\\omega t)')
    elev_line = proto.get('elev_line', '>     0.00')
    origin_line = proto.get('origin_line', '>    0.000   0.000')
    per_set = sorted({r[0] for r in resp.record_order})
    # ModEM 的 n_sites 计站位 (code,x,y)——同码多坐标（网格复用码）时
    # 唯一码数 ≠ 站位数，按站位数才与数据行一致
    sta_n = len(resp.station_keys())
    lines = ['# synthts: EM_Fields placeholder (zeros; modified ModEM fills'
             ' after forward)',
             '# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error',
             '>  EM_Fields',
             exp_line,
             '>  [mV/km]/[nT]',
             elev_line,
             origin_line,
             '>%8d%6d' % (len(per_set), sta_n)]
    # 注意：改版 exe 按固定结构分段——每段恰好 2 行 '#' 注释 + 6 行 '>' 头
    # （实测：多出的注释行会被当成数据类型声明，报 Unknown data type）
    for per, code, x, y in resp.record_order:
        # 周期/坐标 token 原文照抄：%.6E 重排会舍入全精度周期
        # （如 7281.73013908105），与源行的浮点键错位产生幻影记录
        tok = resp.raw_row.get((per, code, x, y))
        if tok:
            head_cols = '  '.join([tok[0], tok[1], tok[2], tok[3],
                                   tok[4], tok[5], tok[6]])
        else:
            lat, lon, elev = resp.coords.get((code, x, y),
                                             (0.0, 0.0, 0.0))
            head_cols = '%.6E  %s  %.7f  %.7f  %.3f  %.3f  %.3f' % (
                per, code, lat, lon, x, y, elev)
        for comp in _EM_COMPS:
            lines.append('%s  %-6s  0.000000E+00  0.000000E+00  '
                         '2.000000E+15' % (head_cols, comp))
    from pathlib import Path
    body = ''.join(ln + '\r\n' for ln in lines).encode('ascii')
    src_bytes = Path(src).read_bytes()
    if src_bytes and not src_bytes.endswith((b'\n', b'\r')):
        src_bytes += b'\r\n'          # 源文件缺尾换行时补齐，防注释并入数据行
    Path(dst).write_bytes(src_bytes + body)
    return {'n_periods': len(per_set), 'n_stations': sta_n,
            'n_records': len(resp.record_order),
            'n_rows': len(resp.record_order) * len(_EM_COMPS),
            'sections': [n for n, _, _ in resp.sections]}
