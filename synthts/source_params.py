# -*- coding: utf-8 -*-
"""v2 源参数：预设、校验与 L4 schema 冻结（设计 §4.1/§9.5）。

CLI 参数面克制原则（D3）：仅 -source/-preset/-source-params 三个新参数；
其余源参数经 JSON 文件注入（不含 seed/起点——二者属 CLI 与不变量）。
"""
import json
import os
import sys

# 预设表（M1 仅 pure-l1 激活；natural/full-event 为 M2 排期）
PRESETS = {
    'pure-l1': {'alpha1': 1.0, 'alpha2': 0.0, 'alpha3': 0.0},
    'natural': None,        # M2：L2 分带突发（§2.2 默认参数）
    'full-event': None,     # M2：natural + L3 事件层
}

# 源参数 schema v1（可注入项与取值约束；只增不改，语义版本号见 SCHEMA_VER）
SCHEMA_VER = 1
PARAM_SCHEMA = {
    'alpha1': (float, 0.0, 1.0),        # L1 底权重（平均功率份额）
    'alpha2': (float, 0.0, 1.0),        # L2 带突发权重
    'alpha3': (float, 0.0, 1.0),        # L3 事件层权重
    'band_octave': (float, 1.0 / 3.0, 1.0),   # 带间距（M2）
    'burst_periods': (float, 10.0, 256.0),    # 平均突发时长（周期数，M2）
    'burst_len_sigma': (float, 0.25, 1.0),    # 段长 LN σ（ln 单位，M2）
    'burst_sigma_db': (float, 3.0, 10.0),     # 段增益 LN σ（dB，M2）
    'common_sigma_db': (float, 0.0, 3.0),     # 跨带公共因子 γ σ（M2，0=关）
    'event_rate': (float, 0.0, 1e6),          # L3 事件率（次/记录小时，M2）
    'event_q': (float, 3.0, 100.0),           # L3 Gabor Q（M2）
}

# L4 加性噪声层参数 schema（§9.5 冻结；M4 实现，v2.0 仅占位校验）
L4_SCHEMA = {
    'type': ('white', 'pink', 'powerline', 'impulsive', 'playback'),
    'level_db': (float, -120.0, 0.0),   # 相对通道 RMS 的加性水平
    'spectrum': (str, None),            # 谱形描述（pink/powerline 细项）
    'per_channel': (dict, None),        # 每通道覆盖
    'playback_file': (str, None),       # 实测噪声回放文件（type=playback）
}


def validate(params):
    """参数校验（非法组合启动期断言，§6 风险表）。返回归一化 dict。"""
    if params is None:
        params = {}
    unknown = set(params) - set(PARAM_SCHEMA)
    if unknown:
        raise SystemExit('unknown source params: %s' % sorted(unknown))
    out = dict(params)
    for key, spec in PARAM_SCHEMA.items():
        if key not in out:
            continue
        kind, lo, hi = spec
        v = out[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise SystemExit('param %s must be numeric' % key)
        if not (lo <= v <= hi):
            raise SystemExit('param %s out of range [%s, %s]' % (key, lo, hi))
    total = out.get('alpha1', 0) + out.get('alpha2', 0) + out.get('alpha3', 0)
    if 'alpha1' in out or 'alpha2' in out or 'alpha3' in out:
        if abs(total - 1.0) > 1e-9 and total != 0.0:
            raise SystemExit('alpha1+alpha2+alpha3 must be 1 (got %g)' % total)
    return out


def resolve(preset, params_file):
    """预设 + 注入文件 → 生效参数。返回 (params_dict, preset_name)。"""
    if preset not in PRESETS:
        raise SystemExit('unknown preset: %s' % preset)
    if preset in ('natural', 'full-event'):
        raise SystemExit('preset %s is scheduled for M2 (not implemented)' % preset)
    params = dict(PRESETS[preset]) if PRESETS[preset] else {}
    if params_file:
        with open(params_file, encoding='utf-8') as fh:
            override = json.load(fh)
        if not isinstance(override, dict):
            raise SystemExit('source-params file must be a JSON object')
        params.update(validate(override))
    return params, preset
