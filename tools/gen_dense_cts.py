# -*- coding: utf-8 -*-
"""用 syscal_native（净室逆向纯 Python 引擎，≡原 exe 443 机队位级）
生成稠密 CTS 表 → synthts/cts_dense.py。

2026-11-03 重写（替代 dll+逐 band 对齐版）：
  - 引擎 = mt-calibration/scripts/syscal_native（与 mtproc phoenix/MTDataPro
    原生链同源）——稠密表与处理端**结构上一致**，旧版"逐 band 复数中位
    比例对齐到稀疏表"的 hack 废除（该对齐正是稠密表 vs native 系统性
    偏差的来源，见 docs/parity_mtproc_phoenix_a0caf41.md）。
  - 网格 = 每 band 全计划覆盖（渲染 bin 可出现在 TS_SPEC 全范围，按文件
    采样率路由后各 band 须覆盖其文件的全部计划频率）：
      band2  9-12500 Hz（TS2 10-12000）
      band3  0.9-1250 Hz（TS3 1-1200）
      band4  0.009-80 Hz（TS4 0.1-75；AMT 连续 TS4 到 0.01）
      band5  1.4e-5-7.7 Hz（TS5 到 7.5 = 15 Hz 链 Nyquist）
    底网格 48 点/倍频程 + 自适应加密（相邻锚点相位步 >10° 或幅值比
    离 1 超 5% 时二分，8 轮——解析带缘 knee 与通道零点）+ 50 Hz 工频
    陷波邻域加密 + 稀疏表全部锚点频率。
  - 值 = 引擎原始 z（处处连续平滑；运行时插值后再 transform 为
    responds——T=adc·conj(z)/|z|^2 在 z 零/极点处符号歧义，插值须在
    z 空间做）；生成端不做
    退化行剔除——运行时 calib._dense_group 的 zero/20x 局部离群过滤
    兜底（数据文件保持引擎原始输出）。

用法（仓库根）：python tools/gen_dense_cts.py
"""
import math
import os
import sys

import numpy as np

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBTREE_ROOT = os.path.dirname(_TOOLS_DIR)
# 布局候选：mt-skills（subtree 内，引擎在兄弟 skill）｜上游（引擎经环境变量）
_CANDS = [
    os.path.join(os.path.dirname(_SUBTREE_ROOT), 'mt-calibration', 'scripts'),
    os.environ.get('SYNTHTS_SYSCAL_NATIVE') or '',
]
_ENGINE_DIR = next((c for c in _CANDS
                    if c and os.path.isfile(os.path.join(c, 'syscal_native.py'))),
                   '')
if not _ENGINE_DIR:
    raise SystemExit('syscal_native 引擎未找到（设 SYNTHTS_SYSCAL_NATIVE=引擎目录）；'
                     '试过: %s' % '; '.join(c for c in _CANDS if c))
_ASSETS_DIR = os.path.join(_SUBTREE_ROOT, 'synthts', 'assets')
_OUT_PATH = os.path.join(_SUBTREE_ROOT, 'synthts', 'cts_dense.py')
if not os.path.isdir(_ASSETS_DIR):
    raise SystemExit('资产目录未找到: %s' % _ASSETS_DIR)

sys.path.insert(0, _TOOLS_DIR)
sys.path.insert(0, _ENGINE_DIR)
from synthts import calib                                       # noqa: E402
import syscal_native as sn                                      # noqa: E402

# 每 band 全计划覆盖（渲染按文件采样率路由，见 calib.band_of_sr）
_RANGES = {2: (9.0, 11000.0), 3: (0.9, 1250.0),  # band2 上缘 11.3k+ 出 0x4CD 括号
           4: (0.009, 80.0), 5: (1.4e-5, 7.7)}


def _eval_channel(sess, band, f):
    rws, ff = sess.evaluate([(float(f), band)])
    assert ff == 0
    _f, _b, rc, rec = rws[0]
    if rc != 0:
        return None
    return [calib.transform_cts(z) for z in rec]


def build_grid(sess, band):
    """底网格 48 点/倍频程 + 自适应加密。

    自适应判据（相邻锚点，逐通道）：相位步 >20° 或 log 幅值步离 0 超
    10% → 插中点；6 轮。解析带缘 knee（如 band4 6-11 Hz）与通道零点
    （相位狂转区）在 12/oct 下插值误差达 5-50%，加密后收敛。"""
    lo, hi = _RANGES[band]
    g = []
    f = lo
    while f < hi:                       # 48 点/倍频程
        g.append(f)
        f *= 2 ** (1 / 48.0)
    g.append(hi)
    if band in (3, 4):                  # 50 Hz 工频陷波邻域加密
        g.extend(np.arange(44.0, 56.01, 0.2))
        g.extend(np.arange(49.0, 51.01, 0.05))
    g.extend(fq for fq, b, _ in (calib.CTS_MT + calib.CTS_AMT) if b == band)
    g = sorted({float(x) for x in g if lo * 0.99 <= x <= hi * 1.01})

    vals = {f: _eval_channel(sess, band, f) for f in g}
    g = [f for f in g if vals[f] is not None]
    for _round in range(8):
        inserts = []
        for a, b in zip(g, g[1:]):
            for c in range(5):
                za = vals[a][c]
                zb = vals[b][c]
                ra, rb = abs(za), abs(zb)
                if ra < 1e-300 or rb < 1e-300:
                    inserts.append(0.5 * (a + b))
                    break
                dph = abs(math.atan2(zb.imag, zb.real)
                          - math.atan2(za.imag, za.real))
                dph = math.degrees(min(dph, 2 * math.pi - dph))
                dlog = abs(math.log10(rb / ra))
                if dph > 10.0 or dlog > 0.02:    # 5% 幅值比 / 10° 相位
                    inserts.append(0.5 * (a + b))
                    break
        if not inserts:
            break
        for m in inserts:
            if m not in vals:
                vals[m] = _eval_channel(sess, band, m)
        g = sorted(f for f in vals if vals[f] is not None)
    print('  band%d 自适应后 %d 点（%d 轮加密）' % (band, len(g), _round + 1))
    return g


def main():
    sess = sn.NativeSession()
    code = sess.load(os.path.join(_ASSETS_DIR, 'DEFAULT.TBL'),
                     os.path.join(_ASSETS_DIR, '4360.CLB'),
                     os.path.join(_ASSETS_DIR, 'BMT50000.CLC'))
    assert code == 0, 'native load err=0x%X' % code
    assert sess.channel_count() == 5

    rows = []
    for band in (2, 3, 4, 5):
        grid = build_grid(sess, band)
        n_ok = 0
        for i in range(0, len(grid), 400):
            rws, ff = sess.evaluate([(f, band) for f in grid[i:i + 400]])
            assert ff == 0
            for f, _b, rc, rec in rws:
                if rc != 0:
                    print('  band%d f=%g 求值失败 rc=0x%X（跳过）'
                          % (band, f, rc))
                    continue
                # 存原始 z（引擎输出，处处连续——含极点/零点处无符号
                # 歧义）；responds 语义由运行时插值后 transform_cts 施加
                vals = []
                for z in rec:
                    vals.extend((z.real, z.imag))
                rows.append((f, band, vals))
                n_ok += 1
        print('band%d: %d/%d 点（%g-%g Hz）'
              % (band, n_ok, len(grid), grid[0], grid[-1]))

    with open(_OUT_PATH, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('# -*- coding: utf-8 -*-\n')
        fh.write('"""稠密 CTS 表（tools/gen_dense_cts.py 生成——syscal_native\n')
        fh.write('直接求值，responds 语义 adcFactor*conj(z)/|z|^2，勿手改）。\n')
        fh.write('\n')
        fh.write('行 = (freq_hz, band, [re, im] x 5 通道 Ex,Ey,Hx,Hy,Hz)；\n')
        fh.write('每 band 全计划覆盖（渲染按文件采样率路由，见 calib.band_of_sr）：\n')
        fh.write('band2 9-12500 / band3 0.9-1250 / band4 0.009-80 /\n')
        fh.write('band5 1.4e-5-7.7 Hz；12 点/倍频程 + 50 Hz 陷波加密 + 稀疏锚点。\n')
        fh.write('"""\n')
        fh.write('DENSE_CTS = [\n')
        for f, band, vals in rows:
            cells = ', '.join(repr(v) for v in vals)
            fh.write('    (%r, %d, [%s]),\n' % (f, band, cells))
        fh.write(']\n')
    print('written', os.path.normpath(_OUT_PATH), 'rows=%d' % len(rows))


if __name__ == '__main__':
    main()
