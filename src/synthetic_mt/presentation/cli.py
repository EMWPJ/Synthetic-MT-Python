#!/usr/bin/env python3
"""
SyntheticMT - 大地电磁合成时间序列生成工具

命令行接口
"""

import argparse
import sys
from datetime import datetime
from synthetic_mt import (
    SyntheticTimeSeries, SyntheticSchema, SyntheticMethod, create_test_site,
    NoiseType, NoiseConfig, NoiseInjector, load_modem_file,
    SYNTHETIC_METHOD_NAMES, TS_CONFIGS
)
from phoenix import TsnFile, TblFile
from synthetic_mt import ClbFile, ClcFile


def generate_timeseries(args):
    """生成合成时间序列"""
    site = create_test_site()
    schema = SyntheticSchema.from_ts(args.ts_config)
    
    if args.method is not None:
        method = SyntheticMethod(args.method)
    else:
        method = SyntheticMethod.RANDOM_SEG_PARTIAL
    
    synth = SyntheticTimeSeries(schema, method)
    
    begin = datetime.strptime(args.begin_time, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')
    
    ex, ey, hx, hy, hz = synth.generate(begin, end, site, seed=args.seed)
    
    print(f'Generated {len(ex)} samples ({args.ts_config})')
    print(f'Ex range: [{ex.min():.4f}, {ex.max():.4f}] V/m')
    print(f'Ey range: [{ey.min():.4f}, {ey.max():.4f}] V/m')
    print(f'Hx range: [{hx.min():.6f}, {hx.max():.6f}] A/m')
    print(f'Hy range: [{hy.min():.6f}, {hy.max():.6f}] A/m')
    print(f'Hz range: [{hz.min():.6f}, {hz.max():.6f}] A/m')
    
    if args.output:
        import numpy as np
        data = np.column_stack([ex, ey, hx, hy, hz])
        header = 'Ex(V/m),Ey(V/m),Hx(A/m),Hy(A/m),Hz(A/m)'
        np.savetxt(args.output, data, delimiter=',', header=header)
        print(f'Saved to {args.output}')


def list_methods(args):
    """列出所有合成方法"""
    print('Available synthesis methods:')
    for method in SyntheticMethod:
        print(f'  {method.value}: {SYNTHETIC_METHOD_NAMES[method]}')


def list_ts_configs(args):
    """列出所有TS配置"""
    print('Available TS configurations:')
    for name, config in TS_CONFIGS.items():
        print(f'  {name}:')
        print(f'    Sample rate: {config["sample_rate"]} Hz')
        print(f'    Frequency range: {config["freq_min"]} - {config["freq_max"]} Hz')


def read_tsn(args):
    """读取TSn文件"""
    tsn = TsnFile()
    data, tags = tsn.load(args.input)
    parsed_tags, times = tsn.parse_tags(tags)
    
    print(f'Loaded {len(data)} samples from {args.input}')
    print(f'Channels: {parsed_tags["channel"][0]}')
    print(f'Sample rate: {parsed_tags["sample_rate"][0]} Hz')
    print(f'Scans per record: {parsed_tags["scans"][0]}')
    print(f'Time range: {times[0]} to {times[-1]}')


def read_tbl(args):
    """读取TBL文件"""
    tbl = TblFile(args.input)
    
    print(f'Loaded TBL file: {args.input}')
    print('Configuration:')
    for key in tbl.keys():
        print(f'  {key}: {tbl[key]}')


def main():
    parser = argparse.ArgumentParser(
        description='SyntheticMT - 大地电磁合成时间序列生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate time series with default settings
  python -m src.cli generate --begin-time "2023-01-01 00:00:00" --end-time "2023-01-01 00:01:00"

  # List all synthesis methods
  python -m src.cli methods

  # List TS configurations
  python -m src.cli configs

  # Read a TSn file
  python -m src.cli read-tsn input.tsn
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic time series')
    gen_parser.add_argument('--begin-time', required=True, help='Start time (YYYY-MM-DD HH:MM:SS)')
    gen_parser.add_argument('--end-time', required=True, help='End time (YYYY-MM-DD HH:MM:SS)')
    gen_parser.add_argument('--ts-config', default='TS3', choices=list(TS_CONFIGS.keys()), help='TS configuration')
    gen_parser.add_argument('--method', type=int, choices=[m.value for m in SyntheticMethod], help='Synthesis method')
    gen_parser.add_argument('--seed', type=int, help='Random seed')
    gen_parser.add_argument('--output', help='Output file (CSV format)')
    gen_parser.set_defaults(func=generate_timeseries)
    
    methods_parser = subparsers.add_parser('methods', help='List synthesis methods')
    methods_parser.set_defaults(func=list_methods)
    
    configs_parser = subparsers.add_parser('configs', help='List TS configurations')
    configs_parser.set_defaults(func=list_ts_configs)
    
    tsn_parser = subparsers.add_parser('read-tsn', help='Read TSn file')
    tsn_parser.add_argument('input', help='Input TSn file')
    tsn_parser.set_defaults(func=read_tsn)
    
    tbl_parser = subparsers.add_parser('read-tbl', help='Read TBL file')
    tbl_parser.add_argument('input', help='Input TBL file')
    tbl_parser.set_defaults(func=read_tbl)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())