# -*- coding: utf-8 -*-
"""【已废弃】旧实验：稠密 CTS 对"最近邻标定台阶地板"的改善。

2026-11-23 起标定必须是频率的连续函数（铁律）：最近邻阶梯标定已从 synthts 删除
（`calib.SMOOTH_RESPONDS=False` 分支与 mtprocess 的台阶掩码一并删除），本实验
对照的对象不复存在。稠密表的生成见 `tools/gen_dense_cts.py`（syscal_native 逐点
求值）；本文件的历史版本见 git 记录。
"""
raise SystemExit(
    "已废弃（2026-11-23）：本实验对照的是'内嵌 50 点最近邻表'与稠密表，"
    "最近邻阶梯标定已从 synthts 删除（标定必须是频率的连续函数）。"
    "稠密表的生成见 tools/gen_dense_cts.py。")
