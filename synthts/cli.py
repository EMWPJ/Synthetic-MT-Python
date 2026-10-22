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
        description='Phoenix MT 合成时间序列（RandomSegmentPartiallyWindowed，内置标定）')
    p.add_argument('-model', required=True, nargs='+', metavar='XMLSITE',
                   help='一个或多个 .XMLSITE 模型（批量多测点，同频点共享同源随机源）')
    p.add_argument('-mode', default='MT', choices=['MT', 'AMT'],
                   help='合成模式（MT→TS3/4/5，AMT→TS2/3/4，默认 MT）')
    p.add_argument('-out', required=True, help='输出目录')
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
    p.add_argument('-source', default='lines', choices=['lines', 'noise'],
                   help='合成引擎：lines=v1 线谱（默认，逐字节不变）；'
                        'noise=v2 噪声源（M1：L1 连续谱底）')
    p.add_argument('-preset', default='natural',
                   help='noise 源预设（natural 默认=分层带突发；pure-l1；'
                        'full-event=+事件层）')
    p.add_argument('-source-params', default='', metavar='FILE.json',
                   help='noise 源参数覆盖 JSON（schema 见 synthts/source_params.py；'
                        '不含 seed/起点）')
    return p.parse_args(argv)


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
            children = noise.schema_children([p.name for p in plans.values()])
            n_bands_info = {}
            for schema in schemas:
                t_seg = time.time()
                plan = plans[schema['name']]
                ts_list, info = noise.render_schema(
                    [site for _, site in sites], schema, plan,
                    children[schema['name']], min(sig_all), max(sig_all), dur)
                n_bands_info[schema['name']] = info['n_bands']
                events_dropped += info['events_dropped']
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
        say('ERROR: %s' % e)
        import traceback
        if os.environ.get('SYNTHTS_DEBUG'):
            traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
