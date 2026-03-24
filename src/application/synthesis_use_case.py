"""Application layer - Synthesis Use Case.

The use case orchestrates the synthesis process by coordinating domain services
and infrastructure to fulfill the user's request to synthesize MT time series.
"""

from datetime import datetime
from typing import Optional

import numpy as np

from ..domain.services.synthesis import (
    SyntheticSchema,
    SyntheticTimeSeries,
    load_modem_file,
    calculate_mt_scale_factors,
    create_test_site,
)
from ..domain.services.noise import NoiseInjector
from ..domain.value_objects import SyntheticMethod, NoiseConfig, TS_CONFIGS

from .dto import SynthesisRequest, SynthesisResult, OutputFormat


class SynthesisUseCase:
    """Main use case for magnetotelluric time series synthesis.
    
    This use case orchestrates the full synthesis workflow:
    1. Load forward modeling data from ModEM format file
    2. Calculate MT scale factors based on natural magnetic field amplitude
    3. Generate synthetic time series using specified method
    4. Optionally add noise to the generated time series
    5. Return result with metadata
    
    Example:
        >>> request = SynthesisRequest(
        ...     modem_path='model.dat',
        ...     ts_config='TS3',
        ...     method=SyntheticMethod.RANDOM_SEG_PARTIAL,
        ...     output_format=OutputFormat.NUMPY,
        ...     seed=42
        ... )
        >>> use_case = SynthesisUseCase()
        >>> result = use_case.execute(request)
        >>> print(f"Generated {len(result.ex)} samples")
    """

    def __init__(self):
        """Initialize the synthesis use case."""
        pass

    def execute(self, request: SynthesisRequest) -> SynthesisResult:
        """Execute the synthesis workflow.
        
        Args:
            request: Synthesis request containing all parameters
            
        Returns:
            SynthesisResult with generated time series and metadata
            
        Raises:
            FileNotFoundError: If modem_path file does not exist
            ValueError: If ts_config is not a valid configuration name
        """
        # Validate ts_config
        if request.ts_config not in TS_CONFIGS:
            raise ValueError(
                f"Invalid ts_config '{request.ts_config}'. "
                f"Valid options: {list(TS_CONFIGS.keys())}"
            )
        
        # Load forward modeling data
        sites = load_modem_file(request.modem_path)
        if not sites:
            raise ValueError(f"No sites found in ModEM file: {request.modem_path}")
        
        site = sites[0]
        
        # Create schema and synthesizer
        schema = SyntheticSchema.from_ts(request.ts_config)
        method = request.method if request.method is not None else SyntheticMethod.RANDOM_SEG_PARTIAL
        synthesizer = SyntheticTimeSeries(schema, method)
        
        # Calculate MT scale factors
        scale_e, scale_b = calculate_mt_scale_factors(site)
        site.update_nature_magnetic_amplitude(scale_e, scale_b)
        
        # Generate time series
        # Use a default time window (can be extended to accept time parameters)
        begin_time = datetime(2023, 1, 1, 0, 0, 0)
        end_time = datetime(2023, 1, 1, 0, 1, 0)  # 1 minute
        
        ex, ey, hx, hy, hz = synthesizer.generate(
            begin_time, end_time, site, seed=request.seed
        )
        
        # Add noise if configured
        if request.noise_config is not None:
            injector = NoiseInjector(
                request.noise_config, 
                schema.sample_rate,
                seed=request.seed
            )
            ex, ey, hx, hy, hz = injector.add_noise(ex, ey, hx, hy, hz)
        
        # Calculate duration
        duration = (end_time - begin_time).total_seconds()
        
        # Build metadata
        metadata = {
            'site_name': site.name,
            'site_x': site.x,
            'site_y': site.y,
            'ts_config': request.ts_config,
            'method': method.name if method else 'RANDOM_SEG_PARTIAL',
            'n_frequencies': len(site.fields),
            'frequency_range': (
                float(np.min(site.frequencies())),
                float(np.max(site.frequencies()))
            ),
            'output_format': request.output_format.value,
        }
        
        if request.noise_config is not None:
            metadata['noise'] = {
                'type': request.noise_config.noise_type.name,
                'amplitude': request.noise_config.amplitude,
                'frequency': request.noise_config.frequency,
            }
        
        return SynthesisResult(
            ex=ex,
            ey=ey,
            hx=hx,
            hy=hy,
            hz=hz,
            sample_rate=float(schema.sample_rate),
            duration=duration,
            metadata=metadata,
        )

    def execute_with_site(
        self,
        site,
        ts_config: str = 'TS3',
        method: Optional[SyntheticMethod] = None,
        noise_config: Optional[NoiseConfig] = None,
        begin_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        seed: Optional[int] = None,
    ) -> SynthesisResult:
        """Execute synthesis with an existing ForwardSite.
        
        This variant accepts a pre-loaded ForwardSite object directly,
        useful when you already have the site data loaded.
        
        Args:
            site: Pre-loaded ForwardSite object
            ts_config: Time series configuration name
            method: Synthesis method (uses default if None)
            noise_config: Optional noise configuration
            begin_time: Start time (defaults to 2023-01-01 00:00:00)
            end_time: End time (defaults to 1 minute after begin_time)
            seed: Random seed for reproducibility
            
        Returns:
            SynthesisResult with generated time series
        """
        # Validate ts_config
        if ts_config not in TS_CONFIGS:
            raise ValueError(
                f"Invalid ts_config '{ts_config}'. "
                f"Valid options: {list(TS_CONFIGS.keys())}"
            )
        
        # Create schema and synthesizer
        schema = SyntheticSchema.from_ts(ts_config)
        synth_method = method if method is not None else SyntheticMethod.RANDOM_SEG_PARTIAL
        synthesizer = SyntheticTimeSeries(schema, synth_method)
        
        # Calculate MT scale factors
        scale_e, scale_b = calculate_mt_scale_factors(site)
        site.update_nature_magnetic_amplitude(scale_e, scale_b)
        
        # Default time window
        if begin_time is None:
            begin_time = datetime(2023, 1, 1, 0, 0, 0)
        if end_time is None:
            end_time = datetime(2023, 1, 1, 0, 1, 0)
        
        # Generate time series
        ex, ey, hx, hy, hz = synthesizer.generate(
            begin_time, end_time, site, seed=seed
        )
        
        # Add noise if configured
        if noise_config is not None:
            injector = NoiseInjector(noise_config, schema.sample_rate, seed=seed)
            ex, ey, hx, hy, hz = injector.add_noise(ex, ey, hx, hy, hz)
        
        # Calculate duration
        duration = (end_time - begin_time).total_seconds()
        
        # Build metadata
        metadata = {
            'site_name': site.name,
            'site_x': site.x,
            'site_y': site.y,
            'ts_config': ts_config,
            'method': synth_method.name if synth_method else 'RANDOM_SEG_PARTIAL',
            'n_frequencies': len(site.fields),
            'frequency_range': (
                float(np.min(site.frequencies())),
                float(np.max(site.frequencies()))
            ),
        }
        
        if noise_config is not None:
            metadata['noise'] = {
                'type': noise_config.noise_type.name,
                'amplitude': noise_config.amplitude,
                'frequency': noise_config.frequency,
            }
        
        return SynthesisResult(
            ex=ex,
            ey=ey,
            hx=hx,
            hy=hy,
            hz=hz,
            sample_rate=float(schema.sample_rate),
            duration=duration,
            metadata=metadata,
        )
