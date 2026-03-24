"""Domain services - Cross-cutting domain logic.

Domain services contain business logic that doesn't naturally belong to a single
entity or value object. They coordinate between multiple domain objects and
encapsulate complex operations that are part of the domain model.
"""

from .synthesis import (
    freq_to_time,
    hanning_window,
    inv_hanning_window,
    SyntheticSchema,
    SyntheticTimeSeries,
    calculate_mt_scale_factors,
    create_test_site,
    load_modem_file,
)

__all__ = [
    'freq_to_time',
    'hanning_window',
    'inv_hanning_window',
    'SyntheticSchema',
    'SyntheticTimeSeries',
    'calculate_mt_scale_factors',
    'create_test_site',
    'load_modem_file',
]
