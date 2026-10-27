# -*- coding: utf-8 -*-
"""SynthTS 精简版命令行入口（MT 合成 + RandomSegmentPartiallyWindowed + 内置标定）。

用法:
  python -m synthts.cli -model site.XMLSITE -out OUT_DIR
              [-mode MT|AMT] [-periods 4] [-scale 1] [-rot ex,ey,hx,hy]
              [-dur 秒] [-seed 42] [-log FILE]
              [-source lines|noise] [-preset pure-l1] [-source-params FILE.json]

时间起点固定 2050-01-01 00:00:00（输出 TBL 的 FTIM/LTIM 同步）；
-dur 未指定时按模式缺省：AMT=3600s、MT=72h(259200s)。

TBL/CLB/CLC 与标定全部内置，MT 与 AMT 各一套固定组合（同一 4360 仪器，
采集参数不同；synthts/assets/ 与 synthts/assets/AMT/）——除模型文件外
零外部依赖。其余八算法、噪声支路与 syscal_dll 引擎已移除。

结束时向 stdout 追加一行机器可读摘要：SYNTHTS_SUMMARY {json}
（仅 stdout，不写入 -log 文件；字段见 README/SKILL）。
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

from . import __version__, calib, render_fft, source_params, synth
from .model import load_site, densify_frequencies, resample_site_log, save_site
from .noise_source import NoiseSource
from .phoenix_ts import write_time_series
from .tbl import TBL, from_tdatetime, to_tdatetime


def nature_magnetic_amplitude(fre: float) -> float:
    """FreToTime.NatureMagneticAmplitude（自然磁场幅值 A/m）。"""
    import math
    a1, b1, c1 = 1.846, 0.3578, -2.302
    a2, b2, c2 = 0.4855, 1.466, -0.6232
    a3, b3, c3 = 0.07102, 2.435, -2.001
    a4, b4, c4 = 0.05518, 3.765, -0.8136
    x = math.log10(fre)
    x = max(min(x, 4.0), -4.0)
    r = (a1 * math.sin(b1 * x + c1) + a2 * math.sin(b2 * x + c2)
         + a3 * math.sin(b3 * x + c3) + a4 * math.sin(b4 * x + c4))
    return 10.0 ** r / (400.0 * math.pi) / 10.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='synthts',
        description='Phoenix MT/AMT 合成时间序列（双引擎，内置标定）。\n'
                    '七种工作模式（互斥）：\n'
                    '  -model XMLSITE...      常规合成（XMLSITE 正演模型 → TS2-5/TBL）\n'
                    '  -modem-run MODEL DAT   ModEM 一键：补场值段→改版 exe 正演→'
                    '测点→时序\n'
                    '  -modem DAT             ModEM 响应/场值 → XMLSITE 测点（不合成）\n'
                    '  -modem-add-fields DAT  常规 ModEM dat 补 EM_Fields 占位段\n'
                    '  -edi FILE.EDI...       实测 EDI → XMLSITE 测点（非标准流程，'
                    '强制警告）\n'
                    '  -qc DIR                合成结果 QC：逐站 ρa/φ 理论 vs 合成图\n'
                    '  -info / -selftest      频段与体积速查 / 安装自检',
        epilog='示例：\n'
               '  synthts -model examples/mta.XMLSITE -out out_mta\n'
               '  synthts -modem-run model.modem data.dat -out out1 '
               '-modem-exe ModEM.exe -max-periods 8 -dur 3600\n'
               '  synthts -modem response.dat -out sites/\n'
               '  synthts -edi JHMT03SB.EDI -out sites2/ -dense 4\n'
               '  synthts -qc out_mta            # 对合成输出目录出 QC 图\n'
               '  synthts -info && synthts -selftest\n'
               '机器可读摘要：stdout 最后一行 SYNTHTS_SUMMARY {json}',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-model', nargs='+', metavar='XMLSITE',
                   help='一个或多个 .XMLSITE 正演模型（批量多测点，同频点共享'
                        '同源随机源，远参考场景）')
    p.add_argument('-modem', default='', metavar='DAT',
                   help='ModEM 数据/响应文件（Full_Impedance / Rho_Phase / '
                        'Off_Diagonal_Rho_Phase / Full_Vertical_Components / '
                        'EM_Fields 多段）→ 转合成测点逐站写 <站名>.XMLSITE 到 '
                        '-out（不合成时序）')
    p.add_argument('-modem-run', nargs=2, metavar=('MODEL', 'DAT'),
                   help='ModEM 一键全流程：常规 dat 取子集（-stations/'
                        '-max-periods）→ 补 EM_Fields 占位段 → 短 ASCII 工作'
                        '目录跑改版 exe（-modem-exe）→ 场值转测点 → 紧接着'
                        '透传 -mode/-dur/-seed 等直接合成时序到 -out')
    p.add_argument('-modem-exe', default='', metavar='EXE',
                   help='改版 ModEM exe 路径（-modem-run 必填；exe 命令行'
                        '装不下长中文路径，本流程自动拷贝到短 ASCII 工作目录）')
    p.add_argument('-modem-par', default='', metavar='PAR',
                   help='ModEM 正演参数文件（缺省自动生成紧容差参数）')
    p.add_argument('-stations', default='', metavar='A,B,C',
                   help='-modem-run 测站码子集（缺省全部；影响正演耗时）')
    p.add_argument('-max-periods', type=int, default=None, metavar='N',
                   help='-modem-run 周期数上限（等距取含首尾；正演耗时'
                        '∝周期数，COMMEMI 级网格约 1 分钟/周期）')
    p.add_argument('-keep-work', action='store_true',
                   help='-modem-run 保留 exe 工作目录与全网格场解 f.E'
                        '（数百 MB；缺省删除，日志与响应仍存 -out/fwd/）')
    p.add_argument('-modem-units', default='si', choices=['si', 'raw'],
                   help='ModEM 测点单位：si=阻抗/场值换算到 SI（默认）；raw=保留'
                        '文件原单位（[mV/km]/[nT]）')
    p.add_argument('-modem-conjugate', default='auto',
                   choices=['auto', 'on', 'off'],
                   help='相位约定：auto=实测 ModEM 与 XMLSITE/EDI 同为 +45° 口径'
                        '，默认直通（仅段头 exp(-i 才共轭）；on/off 强制')
    p.add_argument('-modem-add-fields', default='', metavar='DAT',
                   help='常规 ModEM dat → 补 EM_Fields 占位段（EPol*/BPol* 十行、'
                        '初值 0）写新 dat，供改版 ModEM 正演回填场值；-out 为输出'
                        ' dat 路径')
    p.add_argument('-edi', nargs='+', metavar='EDI',
                   help='实测 EDI（SEG 1.0）→ XMLSITE 测点（不合成时序）。'
                        '⚠ 非标准合成流程：合成"理论"=该站实测响应复刻，'
                        '不可作该站无偏真值基准（强制弹出警告）')
    p.add_argument('-qc', nargs='?', const='', metavar='DIR',
                   help='QC 报告：对合成输出目录逐测点出 ρa/φ 理论 vs '
                        '合成对比图（qc_<站>.png）与 qc_summary.json。'
                        '目录可直接跟在 -qc 后，或用 -out 提供')
    p.add_argument('-info', action='store_true',
                   help='打印频段表（TS2-5 采样率/覆盖/连续性）、体积预估、'
                   '标定资产与运行环境速查后退出')
    p.add_argument('-selftest', action='store_true',
                   help='安装自检：临时目录双引擎微型合成 + 锚点阻抗回验'
                        '（约 1 分钟）后退出')
    p.add_argument('-mode', default='MT', choices=['MT', 'AMT'],
                   help='合成模式（MT→TS3/4/5，AMT→TS2/3/4，默认 MT；'
                        'AMT 的 TS2 为 24 kHz，注意体积）')
    p.add_argument('-out', default='', metavar='DIR|FILE',
                   help='输出目录（-modem-add-fields 时为输出 dat 文件路径；'
                        '-qc 时为待检查的合成输出目录）')
    p.add_argument('-periods', type=float, default=4.0, help='平均段长（信号周期数，默认 4）')
    p.add_argument('-scale', type=float, default=1.0, help='幅值系数')
    p.add_argument('-rot', default='0,0,0,0', help='exDeg,eyDeg,hxDeg,hyDeg')
    p.add_argument('-dur', type=float, default=None,
                   help='合成时长秒（未指定时按模式缺省：AMT=3600，MT=72h）')
    p.add_argument('-seed', type=int, default=42, help='随机种子（numpy Generator，默认 42，可复现）')
    p.add_argument('-dense', type=float, default=4.0, metavar='N',
                   help='对数域频点自动加密：每倍频程 N 个点（锚点保留，场按 log 域插值；'
                        '默认 4，0 关闭）')
    p.add_argument('-log', default='', help='运行日志文件')
    p.add_argument('-source', default='noise', choices=['lines', 'noise'],
                   help='合成引擎：noise=v2 三层噪声源（默认，v2.0 起）；'
                        'lines=v1 线谱（永久保留，逐字节不变）')
    p.add_argument('-preset', default='natural',
                   help='noise 源预设（natural 默认=分层带突发；pure-l1；'
                        'full-event=+事件层）')
    p.add_argument('-source-params', default='', metavar='FILE.json',
                   help='noise 源参数覆盖 JSON（schema 见 synthts/source_params.py；'
                        '不含 seed/起点）')
    return p.parse_args(argv)


def _run_modem_run(args, say, t0):
    """-modem-run：子集→补段→exe 正演→测点，再透传参数直接合成时序。"""
    from .modem_run import orchestrate
    if not args.out:
        raise ValueError('-out <目录> 必填（测点 XMLSITE 与 TS 都写到这里）')
    if not args.modem_exe:
        raise ValueError('-modem-run 需要 -modem-exe <改版 ModEM exe 路径>'
                         '（EMWPJ 仓库 ModEM.exe，支持 EM_Fields 回填）')
    model, dat = args.modem_run
    say('SynthTS(Python) — ModEM 一键：模型 + 常规 dat → 正演 → 测点 → 时序')
    say('  model : %s' % model)
    say('  dat   : %s' % dat)
    say('  exe   : %s' % args.modem_exe)
    if args.max_periods:
        say('  warn  : 正演耗时 ∝ 周期数（本机 COMMEMI 级网格约 1 分钟/周期）')
    stations = [s.strip() for s in args.stations.split(',') if s.strip()] \
        or None
    paths, meta = orchestrate(args.modem_exe, model, dat, args.out,
                              stations=stations,
                              max_periods=args.max_periods,
                              keep_work=args.keep_work,
                              par_file=args.modem_par or None, say=say)
    import json as _json
    from pathlib import Path
    meta_path = Path(args.out) / 'modem_run_meta.json'
    meta_path.write_text(_json.dumps(
        {'model': model, 'dat': dat, 'exe': args.modem_exe,
         'stations': stations, **meta}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    say('  meta  : %s' % str(meta_path).replace('\\', '/'))
    # 合成阶段：透传用户参数（内层 main 自带同款缺省）
    inner = ['-model'] + paths + ['-out', args.out]
    for attr, flag in (('mode', '-mode'), ('periods', '-periods'),
                       ('scale', '-scale'), ('rot', '-rot'), ('seed', '-seed'),
                       ('dense', '-dense'), ('source', '-source'),
                       ('preset', '-preset')):
        inner += [flag, str(getattr(args, attr))]
    if args.dur is not None:
        inner += ['-dur', str(args.dur)]
    if args.source_params:
        inner += ['-source-params', args.source_params]
    if args.log:
        inner += ['-log', args.log]
    say('── 合成阶段（-mode %s -dur %s -seed %d …）──'
        % (args.mode, args.dur if args.dur is not None else '缺省',
           args.seed))
    return main(inner)


def _run_edi(args, say, t0):
    """-edi：实测 EDI → XMLSITE 测点（非标准流程，强制警告）。"""
    from .edi import EDI_WARNING, load_edi, site_from_edi
    from .modem import write_modem_sites
    if not args.out:
        raise ValueError('-out <目录> 必填（逐站写 <站名>.XMLSITE）')
    say(EDI_WARNING)
    sites = []
    for p in args.edi:
        edi = load_edi(p)
        name = os.path.splitext(os.path.basename(p))[0]
        site = site_from_edi(edi, name=name, densify=args.dense)
        sites.append(site)
        say('  edi   : %s → %s（%d 频点，%.1f/倍频程加密）'
            % (p, site.name, len(site.fields), args.dense))
    files = write_modem_sites(sites, args.out)
    for f in files:
        say('  site  : %s' % f)
    say('DONE (%.1f s)' % (time.time() - t0))
    summary = {
        'tool': 'synthts', 'version': __version__, 'action': 'edi2xmlsite',
        'warning': '非标准合成流程：EDI 实测阻抗复刻，不可作该站无偏真值基准',
        'edi': args.edi, 'dense': args.dense,
        'sites': [s.name for s in sites], 'n_sites': len(sites),
        'n_freqs': [len(s.fields) for s in sites],
        'out_dir': os.path.abspath(args.out),
        'files': [f.replace('\\', '/') for f in files],
        'n_files': len(files), 'summary_v': 2,
        'elapsed_s': round(time.time() - t0, 3),
    }
    print('SYNTHTS_SUMMARY ' + json.dumps(summary, ensure_ascii=False))
    return 0


def _run_qc(args, say, t0):
    """-qc：合成输出目录 → 逐站 ρa/φ 理论 vs 合成 图表与 JSON。"""
    from .qc import qc_directory
    qc_dir = args.qc or args.out
    if not qc_dir:
        raise ValueError('-qc 需要目录（synthts -qc <合成输出目录>，'
                         '含 *_dense.XMLSITE 与 TS 文件）')
    say('SynthTS(Python) — 合成结果 QC（视电阻率/相位 理论 vs 合成）')
    summary = qc_directory(qc_dir, say=say)
    say('DONE (%.1f s)' % (time.time() - t0))
    med = {}
    for name, s in summary['sites'].items():
        med[name] = {ij: round(s['metrics'][ij]['rho_med_rel'], 4)
                     for ij in ('XY', 'YX')}
    print('SYNTHTS_SUMMARY ' + json.dumps({
        'tool': 'synthts', 'version': __version__, 'action': 'qc',
        'data_dir': summary['data_dir'], 'n_sites': len(summary['sites']),
        'rho_med_rel': med,
        'qc_json': (summary['data_dir'] + '/qc_summary.json'),
        'summary_v': 2, 'elapsed_s': round(time.time() - t0, 3)},
        ensure_ascii=False))
    return 0


def _run_info(say):
    """-info：频段表/体积预估/环境速查（只读退出）。"""
    import platform
    freqs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    n_resp = len(calib.compute_responds(freqs)[0])
    say('SynthTS(Python) v%s — 环境与频段速查' % __version__)
    say('  python %s / numpy %s / 平台 %s'
        % (platform.python_version(), np.__version__, platform.platform()))
    say('  标定 : 内置 CTS 双表，responds 抽验 %d 频点正常（%s）'
        % (n_resp, calib.ASSETS_DIR.replace('\\', '/')))
    tbl = TBL.load(os.path.join(calib.ASSETS_DIR, 'DEFAULT.TBL'))
    for mode in ('MT', 'AMT'):
        say('  %s 档表（-periods 4 默认）：' % mode)
        for s in synth.build_schemas(tbl, mode, 4.0, 1.0):
            kind = '连续' if s['continuous'] else \
                '间歇(%gs 段/%gs 间隔)' % (s['seg_s'], s['interval_s'])
            say('    %-4s %7g Hz  %.4g-%7.4g Hz  %s'
                % (s['name'], s['sample_rate'], s['min_fre'], s['max_fre'],
                   kind))
    say('  体积 : MT ≈ 250 MB/天/站；AMT(TS2 24kHz) ≈ 430 MB/小时/站')
    say('  自检 : synthts -selftest（双引擎微型合成 + 锚点回验，约 1 分钟）')
    say('  详情 : synthts -h / SKILL.md / AGENTS.md')
    return 0


def _run_selftest(say, t0):
    """-selftest：临时目录双引擎微型合成 + 锚点阻抗回验。"""
    import tempfile
    from .model import Field, ForwardSite, save_site
    from . import mtprocess
    say('SynthTS(Python) — 安装自检（双引擎微型合成）')
    rho = 100.0
    freqs = np.logspace(-3, np.log10(320.0), 25)
    ok_all = True
    with tempfile.TemporaryDirectory(prefix='synthts_st_') as td:
        site = ForwardSite()
        site.name = 'SELFTEST'
        for f in freqs:
            z = np.sqrt(2j * np.pi * f * 4e-7 * np.pi * rho)
            fld = Field(frequency=float(f), is_noise=False)
            fld.ex1, fld.ey1 = 0.0j, -z          # Z=[[0,z],[-z,0]]
            fld.ex2, fld.ey2 = z, 0.0j
            fld.hx1, fld.hy1 = 1.0 + 0j, 0.0j
            fld.hx2, fld.hy2 = 0.0j, 1.0j
            site.fields.append(fld)
        model = os.path.join(td, 'selftest.XMLSITE')
        save_site(site, model)
        results = {}
        for source in ('lines', 'noise'):
            out = os.path.join(td, 'out_' + source).replace('\\', '/')
            rc = main(['-model', model, '-out', out, '-mode', 'MT',
                       '-dur', '1800', '-seed', '42', '-source', source])
            if rc != 0:
                say('  [%s] FAIL：合成退出码 %d' % (source, rc))
                ok_all = False
                continue
            n_ts = len(glob.glob(os.path.join(out, '*.TS?')))
            z_th = np.sqrt(2j * np.pi * 1.0 * 4e-7 * np.pi * rho)
            rho_th = abs(z_th) ** 2 / (2 * np.pi * 1.0 * 4e-7 * np.pi)
            r = mtprocess.estimate_impedance_at(1.0, out, 'SELFTEST',
                                                bypass_screen=True)
            if r is None:
                say('  [%s] FAIL：f=1Hz 无覆盖频段（TS 文件 %d 个）'
                    % (source, n_ts))
                ok_all = False
                continue
            zest, nwin = r
            rho_e = abs(zest[0, 1]) ** 2 / (2 * np.pi * 4e-7 * np.pi)
            rel = abs(rho_e - rho_th) / rho_th
            tol = 0.25 if source == 'lines' else 0.60
            ok = rel <= tol
            ok_all &= ok
            say('  [%-5s] %s：TS×%d，f=1Hz ρa %.2f Ωm（理论 %.2f，偏差 '
                '%.1f%%，容差 %d%%，%d 窗）'
                % (source, 'PASS' if ok else 'FAIL', n_ts, rho_e, rho_th,
                   100 * rel, 100 * tol, nwin))
            results[source] = {'rho': rho_e, 'rel': rel, 'nwin': nwin,
                               'ok': bool(ok)}
    say('SELFTEST %s (%.1f s)' % ('PASS' if ok_all else 'FAIL',
                                  time.time() - t0))
    print('SYNTHTS_SUMMARY ' + json.dumps(
        {'tool': 'synthts', 'version': __version__, 'action': 'selftest',
         'passed': bool(ok_all), 'engines': results, 'summary_v': 2,
         'elapsed_s': round(time.time() - t0, 3)}, ensure_ascii=False))
    return 0 if ok_all else 2


def _run_modem_add_fields(args, say, t0):
    """-modem-add-fields 分支：常规 dat 补 EM_Fields 占位段（不合成）。"""
    from .modem import add_em_fields
    out = os.path.abspath(args.out)
    if args.out.replace('\\', '/').endswith('/') or os.path.isdir(out):
        raise ValueError('-modem-add-fields 的 -out 须为输出 dat 文件路径，'
                         '不是目录: %r' % args.out)
    say('SynthTS(Python) — ModEM dat 补 EM_Fields 占位段')
    say('  input : %s' % args.modem_add_fields)
    say('  output: %s' % out)
    info = add_em_fields(args.modem_add_fields, out)
    say('  source sections: %s' % ', '.join(info['sections']))
    say('  grid  : %d 周期 × %d 站 = %d 组 → 追加 %d 行（初值 0，误差 2.0E+15）'
        % (info['n_periods'], info['n_stations'], info['n_records'],
           info['n_rows']))
    say('DONE (%.1f s)' % (time.time() - t0))
    summary = {
        'tool': 'synthts', 'version': __version__,
        'action': 'modem_add_fields',
        'input': args.modem_add_fields,
        'output': out.replace('\\', '/'),
        'sections': info['sections'],
        'n_periods': info['n_periods'],
        'n_stations': info['n_stations'],
        'n_records': info['n_records'],
        'n_rows_added': info['n_rows'],
        'summary_v': 2,
        'elapsed_s': round(time.time() - t0, 3),
    }
    print('SYNTHTS_SUMMARY ' + json.dumps(summary, ensure_ascii=False))
    return 0


def _run_modem(args, say, t0):
    """-modem 分支：ModEM 正演响应/场值 → 合成测点（XMLSITE），不合成时序。"""
    from .modem import parse_modem_response, sites_from_response, \
        write_modem_sites
    conj = {'auto': 'auto', 'on': True, 'off': False}[args.modem_conjugate]
    resp = parse_modem_response(args.modem)
    say('SynthTS(Python) — ModEM 响应 → 合成测点')
    say('  modem  : %s' % args.modem)
    say('  sections: %s' % ('; '.join('%s(%s)' % (n, u or 'units?')
                                     for n, u, sgn in resp.sections)))
    if not resp.data:
        raise ValueError('ModEM 文件无数据行: %s' % args.modem)
    say('  path   : %s（%d 周期 %.4g–%.4g s，%d 站）'
        % (resp.conversion_path, len(resp.periods), resp.periods[0],
           resp.periods[-1], len(resp.station_keys())))
    sites = sites_from_response(resp, to_si=(args.modem_units == 'si'),
                                conjugate=conj)
    if not sites:
        comps = sorted(resp.comps_present)
        raise ValueError('无有效测点（每站需 ≥2 个完整频点：全四分量'
                         'ZXX..ZYY，或离对角 RHO*/PHS* 可重建，或已回填'
                         ' EM_Fields；文件分量: %s）' % (comps[:12] or '无'))
    got = {s.name for s in sites}
    expect = set()
    for key in resp.station_keys():
        code, x, y = key
        expect.add(code if len(resp.stations.get(code, [])) == 1
                   else '%s_%d' % (code,
                                   resp.stations[code].index((x, y)) + 1))
    dropped = sorted(expect - got)
    if dropped:
        say('  warn  : 丢弃 %d 站（<2 个完整频点）: %s'
            % (len(dropped), ', '.join(dropped[:10])))
    conj_used = (conj is True if conj != 'auto'
                 else bool(resp.sections)
                 and all(sgn == '-i' for _, _, sgn in resp.sections))
    files = write_modem_sites(sites, args.out)
    for p in files:
        say('  site  : %s' % p)
    say('DONE (%.1f s)' % (time.time() - t0))
    summary = {
        'tool': 'synthts', 'version': __version__,
        'action': 'modem2xmlsite',
        'modem': args.modem,
        'sections': [n for n, _, _ in resp.sections],
        'conversion_path': resp.conversion_path,
        'conjugated': bool(conj_used),
        'units': args.modem_units,
        'n_periods': len(resp.periods),
        'period_range_s': [resp.periods[0], resp.periods[-1]],
        'sites': [s.name for s in sites],
        'n_sites': len(sites),
        'n_freqs': [len(s.fields) for s in sites],
        'out_dir': os.path.abspath(args.out),
        'files': [f.replace('\\', '/') for f in files],
        'n_files': len(files),
        'summary_v': 2,
        'elapsed_s': round(time.time() - t0, 3),
    }
    print('SYNTHTS_SUMMARY ' + json.dumps(summary, ensure_ascii=False))
    return 0


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    if args.log:
        class _Tee:
            def __init__(self, path):
                self.f = open(path, 'a', encoding='utf-8')

            def __call__(self, *a):
                print(*a)
                print(*a, file=self.f)
        say = _Tee(args.log)
    else:
        say = print

    try:
        if args.info:
            return _run_info(say)
        if args.selftest:
            return _run_selftest(say, t0)
        given = [nm for nm, v in (
            ('-model', bool(args.model)), ('-modem-run', bool(args.modem_run)),
            ('-modem', bool(args.modem)), ('-modem-add-fields',
                                           bool(args.modem_add_fields)),
            ('-edi', bool(args.edi)), ('-qc', args.qc is not None)) if v]
        if len(given) > 1:
            raise ValueError('工作模式互斥（同时给了 %s）；见 synthts -h 的'
                             '七种模式' % ' + '.join(given))
        if args.modem_run:
            return _run_modem_run(args, say, t0)
        if args.modem_add_fields:
            return _run_modem_add_fields(args, say, t0)
        if args.modem:
            return _run_modem(args, say, t0)
        if args.edi:
            return _run_edi(args, say, t0)
        if args.qc is not None:
            return _run_qc(args, say, t0)
        if not args.model:
            raise ValueError('须指定一种工作模式：-model <XMLSITE...> / '
                             '-modem-run MODEL DAT / -modem DAT / '
                             '-modem-add-fields DAT / -edi FILE... / -qc DIR '
                             '（或 -info / -selftest）')
        if not args.out:
            raise ValueError('-out <目录> 必填（合成输出目录）')
        if args.mode == 'AMT':
            tbl_path = os.path.join(calib.ASSETS_DIR, 'AMT', 'DEFAULT.TBL')
        else:
            tbl_path = os.path.join(calib.ASSETS_DIR, 'DEFAULT.TBL')
        # 时间起点固定 2050-01-01 00:00:00；时长未指定按模式缺省
        btime_dt = datetime(2050, 1, 1, 0, 0, 0)
        dur = args.dur
        if dur is None:
            dur = 3600.0 if args.mode == 'AMT' else 72.0 * 3600.0
        if dur <= 0:
            raise ValueError('-dur 须为正数')

        say('SynthTS(Python) — Phoenix %s 合成时间序列（内置标定，同源批量）' % args.mode)
        say('  model : %s' % ', '.join(args.model))
        say('  tbl   : %s (%s套内置)' % (tbl_path, args.mode))
        say('  out   : %s' % args.out)
        say('  opt   : mode=%s sites=%d periods=%.4f scale=%.4f dur=%ss'
            % (args.mode, len(args.model), args.periods, args.scale, dur))
        src_params = None
        preset_name = None
        if args.source == 'noise':
            src_params, preset_name = source_params.resolve(
                args.preset, args.source_params)
            say('  source: noise (preset=%s, params=%s)'
                % (preset_name, src_params))
            say('  warn  : -periods/-rot 对 noise 源为惰性参数（已忽略）')
        else:
            say('  source: lines (v1 RandomSegmentPartiallyWindowed)')

        os.makedirs(args.out, exist_ok=True)
        out_files = []                     # 登记全部输出文件，供 JSON 摘要

        tbl = TBL.load(tbl_path)
        sites = []
        seen_names = set()
        for m in args.model:
            site = load_site(m)
            if not site.fields:
                raise ValueError('模型无频点: %s' % m)
            if not site.get_frequencies(False):
                raise ValueError('模型无信号频点（全部 isNoise）: %s' % m)
            bad = [f.frequency for f in site.fields
                   if not all(np.isfinite(getattr(f, c))
                              for c in ('ex1', 'ey1', 'hx1', 'hy1', 'hz1',
                                        'ex2', 'ey2', 'hx2', 'hy2', 'hz2'))]
            if bad:
                raise ValueError(
                    '模型含非有限值(NaN/Inf)的频点（%s 共 %d 个，前 5 个: %s）'
                    '——先清洗模型（正演输出常见：粗网格最短周期不可分辨）'
                    % (m, len(bad), ['%.4g' % b for b in bad[:5]]))
            if args.dense > 0:
                n_anchor = len(site.get_frequencies(False))
                dense_freqs = densify_frequencies(site.get_frequencies(False),
                                                  args.dense)
                if len(dense_freqs) > n_anchor:
                    site = resample_site_log(site, dense_freqs)
                    dense_path = os.path.join(args.out, (site.name or 'site')
                                              + '_dense.XMLSITE')
                    save_site(site, dense_path)
                    out_files.append(dense_path.replace('\\', '/'))
                    say('  dense : %s 加密 %d -> %d 频点（%.1f/倍频程, '
                        '场 log 域插值, 模型 %s）'
                        % (site.name, n_anchor, len(dense_freqs), args.dense,
                           dense_path))
            name = site.name or os.path.splitext(os.path.basename(m))[0]
            if name in seen_names:
                raise ValueError('测点名重复（输出文件会互相覆盖）: %s' % name)
            seen_names.add(name)
            sites.append((m, site))

        # 1. 内置标定（CTS_MT/CTS_AMT 双表，band 组内最近邻）。
        # responds 按各测点自身频率序计算（逐频独立，值与并集无异）——
        # add_calibration 按下标对齐场表，用全局并集序会错位
        responds_by_site = {}
        for m, site in sites:
            responds_by_site[m] = calib.compute_responds(site.get_all_frequencies())
        n_union = len({f for _, site in sites for f in site.get_all_frequencies()})
        say('  cal   : 内置CTS双表 %d 频点（%d 测点同源）' % (n_union, len(sites)))

        # 3. 时段与 schema（起点固定，不再取 TBL 的 FTIM；输出 TBL 同步 FTIM/LTIM）
        etime_dt = btime_dt + timedelta(seconds=dur)
        tbl.set_datetime_field('FTIM', btime_dt)
        tbl.set_datetime_field('LTIM', etime_dt)
        btime = to_tdatetime(btime_dt)
        etime = to_tdatetime(etime_dt)
        schemas = synth.build_schemas(tbl, args.mode, args.periods, args.scale)

        import math as _m

        out_dir = args.out.replace('\\', '/')
        if not out_dir.endswith('/'):
            out_dir += '/'

        # 预备：各测点自然幅值缩放 + 名称登记 + TBL 落盘
        site_names = []
        for model_path, site in sites:
            site_name = site.name or os.path.splitext(os.path.basename(model_path))[0]
            site_names.append(site_name)
            # 自然幅值缩放（按各测点自身的极化场归一）
            sig_freqs = site.get_frequencies(False)
            scale_e = [nature_magnetic_amplitude(f) / _m.sqrt(abs(site.fields[i].hx1) ** 2 + abs(site.fields[i].hy1) ** 2)
                       for i, f in enumerate(sig_freqs)]
            scale_b = [nature_magnetic_amplitude(f) / _m.sqrt(abs(site.fields[i].hx2) ** 2 + abs(site.fields[i].hy2) ** 2)
                       for i, f in enumerate(sig_freqs)]
            site.update_nature_magnetic_amplitude(scale_e, scale_b)
            tbl.set_str_field('FILE', site_name)
            tbl.set_str_field('SITE', site_name)
            tbl.save(f'{out_dir}{site_name}.TBL')
            out_files.append(f'{out_dir}{site_name}.TBL')
            say('  输出  : %s%s.TBL' % (out_dir, site_name))

        sig_all = [f for _, site in sites for f in site.get_frequencies(False)]
        n_bands_info = None
        events_dropped = 0
        if args.source == 'noise':
            # v2 噪声源：按记录带合成（设计 §3 方案 (a′)）；标定以逐 bin 最近
            # 邻折进 T（render_fft.transfer_matrix），不走 add_calibration——
            # 与 mtprocess 处理端 compute_responds 的最近邻口径一致
            noise = NoiseSource(args.seed, src_params)
            plans = render_fft.plan_from_schemas(schemas, btime, etime)
            # 全局源结构一次抽签（段链/γ/事件跨档共享，v2.1）
            sr_max = max(p.records[0].sample_rate for p in plans.values()
                         if p.records)
            state = noise.prepare(min(sig_all), max(sig_all), dur, sr_max)
            children = noise.schema_children([p.name for p in plans.values()])
            rec0 = next((r for p in plans.values() for r in p.records), None)
            t0_rec = rec0.begin if rec0 is not None else btime
            record_wins = []
            for plan in plans.values():
                for r in plan.records:
                    record_wins.append(((r.begin - t0_rec) * 86400.0,
                                        (r.begin - t0_rec) * 86400.0
                                        + r.n_samples / r.sample_rate))
            n_bands_info = {}
            events_dropped = 0
            for schema in schemas:
                t_seg = time.time()
                plan = plans[schema['name']]
                ts_list, info = noise.render_schema(
                    [site for _, site in sites], schema, plan,
                    children[schema['name']], state, dur, record_wins)
                n_bands_info[schema['name']] = info['n_bands']
                events_dropped = info['events_dropped']
                for k, ts in enumerate(ts_list):
                    path = write_time_series(ts, out_dir, tbl, site_names[k])
                    out_files.append(path)
                    n_samples = sum(len(s) for s in ts['channels'][0]) if ts['channels'] and ts['channels'][0] else 0
                    say('  synth : %s %s done (%.0f ms, segs=%d samples0=%d) -> %s'
                        % (site_names[k], schema['name'], (time.time() - t_seg) * 1000,
                           len(ts['begins']), n_samples, path))
                    ts_list[k] = None                # 逐测点释放（大数组）
                del ts_list
        else:
            # 5. 同源方案一次抽签 + 逐档整批渲染（载波跨测点复用）+ 逐测点落盘
            rng = np.random.default_rng(args.seed)
            union_freqs = sorted({f for _, site in sites for f in site.get_frequencies(False)})
            scenarios = synth.build_schema_scenarios(schemas, union_freqs, btime,
                                                     etime, args.periods, rng)
            rot = tuple(float(v) * 3.141592653589793 / 180.0 for v in args.rot.split(','))
            if len(rot) != 4:
                raise ValueError('-rot 须为 exDeg,eyDeg,hx,hy')
            import copy

            for schema in schemas:
                t_seg = time.time()
                site_k_list = []
                for model_path, site in sites:
                    site_k = copy.deepcopy(site)
                    site_k.add_calibration(responds_by_site[model_path])
                    site_k_list.append(site_k)
                ts_list = synth.synthetic_sites(site_k_list, schema,
                                                scenarios[schema['name']], rot)
                del site_k_list
                for k, ts in enumerate(ts_list):
                    path = write_time_series(ts, out_dir, tbl, site_names[k])
                    out_files.append(path)
                    n_samples = sum(len(s) for s in ts['channels'][0]) if ts['channels'] and ts['channels'][0] else 0
                    say('  synth : %s %s done (%.0f ms, segs=%d samples0=%d) -> %s'
                        % (site_names[k], schema['name'], (time.time() - t_seg) * 1000,
                           len(ts['begins']), n_samples, path))
                    ts_list[k] = None                # 逐测点释放（大数组）
                del ts_list
        say('DONE (%.1f s)' % (time.time() - t0))
        # 机器可读摘要：仅 stdout（不进 -log），一行 JSON，供脚本/AI 代理解析
        summary = {
            'tool': 'synthts', 'version': __version__, 'mode': args.mode,
            'models': args.model, 'sites': site_names,
            'n_freqs': [len(site.get_frequencies(False)) for _, site in sites],
            'schemas': [s['name'] for s in schemas],
            'begin': btime_dt.isoformat(sep=' '),
            'end': etime_dt.isoformat(sep=' '),
            'dur_s': dur, 'periods': args.periods, 'scale': args.scale,
            'rot': args.rot, 'seed': args.seed, 'dense': args.dense,
            'out_dir': os.path.abspath(args.out),
            'files': out_files, 'n_files': len(out_files),
            'source': args.source,
            'preset': preset_name,
            'source_params': src_params,
            'n_bands': n_bands_info,
            'events_dropped': events_dropped if args.source == 'noise' else None,
            'rng': {'numpy': np.__version__,
                    'threads': os.environ.get('OMP_NUM_THREADS', '')},
            'summary_v': 2,
            'elapsed_s': round(time.time() - t0, 3),
        }
        print('SYNTHTS_SUMMARY ' + json.dumps(summary, ensure_ascii=False))
        return 0
    except Exception as e:
        say('ERROR: %s: %s' % (type(e).__name__, e))
        for h in _error_hints(e):
            say('  提示: %s' % h)
        if not os.environ.get('SYNTHTS_DEBUG'):
            say('  （设置 SYNTHTS_DEBUG=1 显示完整堆栈）')
        else:
            import traceback
            traceback.print_exc()
        return 2


def _error_hints(e):
    """按异常类型/内容给出可操作提示（详细错误信息约定，AGENTS.md）。"""
    hints = []
    if isinstance(e, FileNotFoundError):
        hints.append('文件不存在——检查路径拼写；含中文/超长路径对部分外部'
                     ' exe（ModEM）需先拷到短 ASCII 路径')
    if isinstance(e, IsADirectoryError):
        hints.append('目标是一个目录——该参数此处需要文件路径'
                     '（-modem-add-fields 的 -out 是 dat 文件，不是目录）')
    if isinstance(e, PermissionError):
        hints.append('无写入权限——换输出目录或检查文件占用')
    msg = str(e)
    if '测点名重复' in msg:
        hints.append('同批模型的站名会决定输出文件名；改 XMLSITE 的 '
                     '<Site name=...> 或分批运行')
    if '模型无频点' in msg or '全部 isNoise' in msg:
        hints.append('检查 XMLSITE 是否含 <Fields frequency=... isNoise='
                     '"False"> 行；用 synthts -modem / -edi 转换的测点'
                     '至少需要 2 个完整频点')
    if '无有效测点' in msg:
        hints.append('ModEM 文件分量不足——需要全四分量 ZXX..ZYY、离对角'
                     'RHO*/PHS*（可重建）或已回填的 EM_Fields')
    if '路径不得包含' in msg or '..' in msg and '路径' in msg:
        hints.append('输出/输入路径禁止 .. 上溯（仓库写盘守卫）')
    if 'numpy' in msg.lower():
        hints.append('numpy 版本影响金桶与数值口径——pip 安装与生成环境'
                     '一致的 numpy 版本')
    return hints


if __name__ == '__main__':
    sys.exit(main())
