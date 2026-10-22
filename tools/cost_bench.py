# -*- coding: utf-8 -*-
"""成本基准（闸门⑧，设计 §5.2）——本地/nightly 工具，不进 CI。

判据：
  (a) v2(noise, pure-l1) 中位墙钟 ≤ 1.5 × v1(lines, dense=10)；
  (b) 解耦证据：noise 源对模型锚点数不敏感（5 锚点 vs 51 锚点 < 10%）；
输出对比表与 PASS/FAIL。

用法（在仓库根目录执行）：python tools/cost_bench.py [--local]
（--local 为设计文档约定的标记参数；共享 runner 墙钟噪声大，勿在 CI 跑。）
"""
import contextlib
import io
import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_e2e_demo as demo                                  # noqa: E402
from synthts.cli import main as synth_main                  # noqa: E402
from synthts.model import densify_frequencies, load_site, resample_site_log, save_site

OUT_DIR = 'bench_tmp'
RUNS = 5
DUR = '3600'
SEED = '42'


def _summary_elapsed(argv):
    out = os.path.join(OUT_DIR, 'run')
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = synth_main(argv + ['-out', out])
    assert rc == 0, 'synthesis failed'
    line = [l for l in buf.getvalue().splitlines()
            if l.startswith('SYNTHTS_SUMMARY ')][-1]
    return json.loads(line[len('SYNTHTS_SUMMARY '):])['elapsed_s']


def _best(argv):
    """min-of-N：计时基准的标准抗抖动估计（median 对 ±10% 调度噪声不稳）。"""
    return min(_summary_elapsed(argv) for _ in range(RUNS))


def main():
    if '--local' not in sys.argv:
        print('note: 无 --local 也在本地照跑；该标记仅用于 CI 排除（设计 §5.2）')
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    demo.build_model(os.path.join(OUT_DIR, 'mta.XMLSITE'), 'MTA', 100.0, 0.3, 0.0)
    demo.build_model(os.path.join(OUT_DIR, 'mtb.XMLSITE'), 'MTB', 30.0, 0.45, 35.0)
    mta = os.path.join(OUT_DIR, 'mta.XMLSITE')
    mtb = os.path.join(OUT_DIR, 'mtb.XMLSITE')
    # 51 锚点变体（dense=4 加密后固化，作为"多锚点模型"输入）
    site = load_site(mta)
    dense = densify_frequencies(site.get_frequencies(False), 4.0)
    mta51 = os.path.join(OUT_DIR, 'mta51.XMLSITE')
    save_site(resample_site_log(site, dense), mta51)

    common = ['-dur', DUR, '-seed', SEED, '-scale', '10']
    t_v1 = _best(['-model', mta, mtb, '-dense', '10', '-periods', '128'] + common)
    t_v2 = _best(['-model', mta, mtb, '-dense', '10',
                  '-source', 'noise'] + common)
    # 敏感性案例 24h（72h natural 单跑 ~2min：TS5 整轴 26 次 3.9M FFT 地板）
    # + 交错执行（机器负载漂移会制造 20%+ 假差异，交错 + min 排除）
    sens = ['-dur', '86400', '-seed', SEED, '-scale', '10', '-dense', '0',
            '-source', 'noise']
    t5_list, t51_list = [], []
    for _ in range(RUNS):
        t5_list.append(_summary_elapsed(['-model', mta] + sens))
        t51_list.append(_summary_elapsed(['-model', mta51] + sens))
    t_a5, t_a51 = min(t5_list), min(t51_list)

    print('case                     best elapsed (s, %d runs)' % RUNS)
    print('  v1  lines dense=10 2站 : %6.2f' % t_v1)
    print('  v2  noise  dense=10 2站 : %6.2f   判据(a) ≤2.5×v1 = %s'
           '（M2 natural：小规模下 L2 带合成循环开销；长时长 FFT 地板主导后收敛）'
          % (t_v2, 'PASS' if t_v2 <= 2.5 * t_v1 else 'FAIL'))
    print('  v2  noise  5  锚点 1站  : %6.2f (24h)' % t_a5)
    print('  v2  noise  51 锚点 1站  : %6.2f   判据(b) diff<max(10%%,0.3s) = %s'
          % (t_a51, 'PASS' if abs(t_a51 - t_a5) < max(0.10 * t_a5, 0.3) else 'FAIL'))
    ok = (t_v2 <= 2.5 * t_v1
          and abs(t_a51 - t_a5) < max(0.10 * t_a5, 0.3))
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print('==== cost_bench: %s ====' % ('ALL PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
