# -*- coding: utf-8 -*-
"""v2 三层源（M1：L1 连续谱底）。

SourceProtocol 形态（设计 §3.5）：rng 分级派生（SeedSequence → 按名称排序
的 schema 子序列），源抽签全部在 schema 层与测点无关——同源合同由结构
保证。L2 分带突发包络与 L3 事件层为 M2 排期（source_params 预设已预留）。
"""
import numpy as np

from . import render_fft


class NoiseSource:
    """L1 白噪源（单位 PSD，谱形由映射层 T 上色）。"""

    def __init__(self, seed, params):
        self.seed = int(seed)
        self.alpha1 = float(params.get('alpha1', 1.0))
        total = self.alpha1 + float(params.get('alpha2', 0.0)) \
            + float(params.get('alpha3', 0.0))
        if abs(total - 1.0) > 1e-9:
            raise SystemExit('alpha1+alpha2+alpha3 must be 1')

    def schema_children(self, schema_names):
        """SeedSequence → 按 schema 名称排序派生子序列（§3.4 分级 rng）。"""
        ss = np.random.SeedSequence(self.seed)
        names = sorted(schema_names)
        return dict(zip(names, ss.spawn(len(names))))

    def render_schema(self, site_list, schema, plan, child_seq):
        """渲染一个 TS 档：源跨测点逐 bin 共享，返回 synthetic_sites 同构。"""
        rng = np.random.default_rng(child_seq)
        return render_fft.synthesize_noise(site_list, schema, plan, rng,
                                           alpha1=self.alpha1)
