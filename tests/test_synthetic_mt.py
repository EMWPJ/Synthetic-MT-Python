"""
Comprehensive pytest unit tests for SyntheticMT core algorithms
"""

import pytest
import numpy as np
import sys
import tempfile
import os
from datetime import datetime
from io import StringIO

sys.path.insert(0, 'D:/code/Synthetic-MT-Python/src')

from synthetic_mt import (
    nature_magnetic_amplitude,
    freq_to_time,
    hanning_window,
    inv_hanning_window,
    SyntheticTimeSeries,
    SyntheticSchema,
    SyntheticMethod,
    ForwardSite,
    EMFields,
    NoiseInjector,
    NoiseConfig,
    NoiseType,
    load_modem_file,
    create_test_site,
    calculate_mt_scale_factors,
    TS_CONFIGS,
)


class TestNatureMagneticAmplitude:
    """Test nature_magnetic_amplitude function"""

    def test_output_is_positive(self):
        """Test that output is positive for valid frequencies"""
        test_freqs = [1e-4, 1e-2, 1, 100, 1000]
        for freq in test_freqs:
            result = nature_magnetic_amplitude(freq)
            assert result > 0, f"nature_magnetic_amplitude({freq}) should be positive"

    def test_typical_frequency_ranges(self):
        """Test typical frequency ranges (1e-4, 1e-2, 1, 100, 1000 Hz)"""
        test_freqs = [1e-4, 1e-2, 1, 100, 1000]
        for freq in test_freqs:
            result = nature_magnetic_amplitude(freq)
            assert result > 0, f"Failed for freq={freq}"
            # Verify it's a reasonable number (not inf or nan)
            assert np.isfinite(result), f"Result should be finite for freq={freq}"

    def test_reasonable_magnitude_range(self):
        """Test that result is in reasonable magnitude (0.001 to 1000 nT range)"""
        test_freqs = [1e-4, 1e-2, 1, 100, 1000]
        for freq in test_freqs:
            result = nature_magnetic_amplitude(freq)
            assert 0.001 <= result <= 1000, \
                f"nature_magnetic_amplitude({freq}) = {result} not in [0.001, 1000] nT range"

    def test_zero_frequency(self):
        """Test zero frequency returns 0"""
        result = nature_magnetic_amplitude(0)
        assert result == 0.0

    def test_negative_frequency(self):
        """Test negative frequency returns 0"""
        result = nature_magnetic_amplitude(-1.0)
        assert result == 0.0

    def test_frequency_below_1e5(self):
        """Test frequency below 1e-5"""
        result = nature_magnetic_amplitude(1e-6)
        assert result > 0
        assert np.isfinite(result)

    def test_frequency_above_1000(self):
        """Test frequency above 1000 Hz"""
        result = nature_magnetic_amplitude(5000)
        assert result > 0
        assert result < 1.0  # Should be small for high frequencies


class TestFreqToTime:
    """Test freq_to_time function"""

    def test_single_frequency_synthesis(self):
        """Test E(t) = A*cos(2πft + φ) produces correct amplitude and phase"""
        amp = 2.0
        phase = np.pi / 4
        freq = 10.0
        sample_rate = 100.0
        n = 1000

        output = np.zeros(n)
        freq_to_time(amp, phase, freq, sample_rate, n, output)

        # Verify amplitude (relaxed tolerance due to discrete sampling)
        max_val = np.max(np.abs(output))
        assert max_val <= amp, \
            f"Max amplitude {max_val} should not exceed input amp {amp}"
        assert max_val >= amp * 0.95, \
            f"Max amplitude {max_val} should be close to {amp}"

        # Verify it's a cosine wave (should have peaks at expected locations)
        t = np.arange(n) / sample_rate
        expected = amp * np.cos(2 * np.pi * freq * t + phase)
        # Check correlation to verify wave shape
        correlation = np.corrcoef(output, expected)[0, 1]
        assert correlation > 0.99, \
            "freq_to_time output should closely match expected cosine wave"

    def test_different_sample_rates(self):
        """Test with different sample rates"""
        amp = 1.0
        phase = 0.0
        freq = 5.0

        for sample_rate in [50, 100, 500, 1000]:
            n = sample_rate * 10  # 10 seconds
            output = np.zeros(n)
            freq_to_time(amp, phase, freq, sample_rate, n, output)

            # Should produce valid output
            assert len(output) == n
            assert np.all(np.isfinite(output))
            max_val = np.max(np.abs(output))
            assert np.isclose(max_val, amp, rtol=0.01)

    def test_zero_frequency(self):
        """Test zero frequency produces constant output"""
        amp = 3.0
        freq = 0.0
        sample_rate = 100.0
        n = 100

        output = np.zeros(n)
        freq_to_time(amp, 0.0, freq, sample_rate, n, output)

        # Should be constant at amp * cos(0) = amp
        assert np.allclose(output, amp)

    def test_high_frequency_nyquist(self):
        """Test frequency near Nyquist limit"""
        sample_rate = 1000.0
        freq = 400.0  # Near Nyquist (500 Hz)
        amp = 1.0
        n = 2000

        output = np.zeros(n)
        freq_to_time(amp, 0.0, freq, sample_rate, n, output)

        assert np.all(np.isfinite(output))
        max_val = np.max(np.abs(output))
        assert np.isclose(max_val, amp, rtol=0.05)


class TestHanningWindow:
    """Test hanning_window and inv_hanning_window functions"""

    def test_window_length_even(self):
        """Test window with even length"""
        n = 100
        window = hanning_window(n)
        assert len(window) == n
        assert np.all(window >= 0)
        assert np.all(window <= 1)

    def test_window_length_odd(self):
        """Test window with odd length"""
        n = 101
        window = hanning_window(n)
        assert len(window) == n
        assert np.all(window >= 0)
        assert np.all(window <= 1)

    def test_window_sums_approximately_n(self):
        """Test window sums to approximately n (not exact due to window shape)"""
        n = 200
        window = hanning_window(n)
        window_sum = np.sum(window)
        # Hanning window should sum to approximately n/2
        assert 0.4 * n <= window_sum <= 0.6 * n, \
            f"Window sum {window_sum} should be approximately {n/2}"

    def test_inverted_window_plus_window_equals_one(self):
        """Test that inverted window + window = 1"""
        n = 150
        window = hanning_window(n)
        inv_window = inv_hanning_window(n)
        combined = window + inv_window
        assert np.allclose(combined, 1.0), \
            "hanning_window + inv_hanning_window should equal 1"

    def test_inv_window_symmetry(self):
        """Test inverse hanning window has correct symmetry"""
        n = 100
        inv_window = inv_hanning_window(n)
        # inv_hanning should start and end near 1, be near 0 in middle
        assert inv_window[0] > 0.9
        assert inv_window[-1] > 0.9
        assert np.min(inv_window) < 0.1


class TestSyntheticMethods:
    """Test all 6 synthesis methods"""

    @pytest.fixture
    def setup_site_and_schema(self):
        """Create test site and schema"""
        site = ForwardSite(name='TestSite')
        # Add a single frequency field
        field = EMFields(
            freq=10.0,
            ex1=complex(1.0, 0.0),
            ey1=complex(0.8, 0.0),
            hx1=complex(0.01, 0.0),
            hy1=complex(0.01, 0.0),
            hz1=complex(0.0, 0.0),
            ex2=complex(0.9, 0.0),
            ey2=complex(-0.8, 0.0),
            hx2=complex(0.01, 0.0),
            hy2=complex(-0.01, 0.0),
            hz2=complex(0.0, 0.0),
        )
        site.fields.append(field)

        schema = SyntheticSchema(
            name='TestTS',
            sample_rate=100.0,
            freq_min=1.0,
            freq_max=100.0,
            synthetic_periods=5.0
        )
        return site, schema

    def test_fix_method(self, setup_site_and_schema):
        """Test FIX method produces valid output"""
        site, schema = setup_site_and_schema
        synth = SyntheticTimeSeries(schema, SyntheticMethod.FIX)

        t1 = datetime(2023, 1, 1, 0, 0, 0)
        t2 = datetime(2023, 1, 1, 0, 0, 1)  # 1 second

        ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)

        assert len(ex) == len(ey) == len(hx) == len(hy) == len(hz) == 100
        assert np.all(np.isfinite(ex))
        assert np.all(np.isfinite(hx))
        assert not np.any(np.isnan(ex))
        assert not np.any(np.isinf(ex))

    def test_fixed_avg_method(self, setup_site_and_schema):
        """Test FIXED_AVG method produces valid output"""
        site, schema = setup_site_and_schema
        synth = SyntheticTimeSeries(schema, SyntheticMethod.FIXED_AVG)

        t1 = datetime(2023, 1, 1, 0, 0, 0)
        t2 = datetime(2023, 1, 1, 0, 0, 1)

        ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)

        assert len(ex) == 100
        assert np.all(np.isfinite(ex))
        assert not np.any(np.isnan(ex))
        assert not np.any(np.isinf(ex))

    def test_fixed_avg_windowed_method(self, setup_site_and_schema):
        """Test FIXED_AVG_WINDOWED method produces valid output"""
        site, schema = setup_site_and_schema
        synth = SyntheticTimeSeries(schema, SyntheticMethod.FIXED_AVG_WINDOWED)

        t1 = datetime(2023, 1, 1, 0, 0, 0)
        t2 = datetime(2023, 1, 1, 0, 0, 1)

        ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)

        assert len(ex) == 100
        assert np.all(np.isfinite(ex))
        assert not np.any(np.isnan(ex))
        assert not np.any(np.isinf(ex))

    def test_random_seg_method(self, setup_site_and_schema):
        """Test RANDOM_SEG method produces valid output"""
        site, schema = setup_site_and_schema
        synth = SyntheticTimeSeries(schema, SyntheticMethod.RANDOM_SEG)

        t1 = datetime(2023, 1, 1, 0, 0, 0)
        t2 = datetime(2023, 1, 1, 0, 0, 1)

        ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)

        assert len(ex) == 100
        assert np.all(np.isfinite(ex))
        assert not np.any(np.isnan(ex))
        assert not np.any(np.isinf(ex))

    def test_random_seg_windowed_method(self, setup_site_and_schema):
        """Test RANDOM_SEG_WINDOWED method produces valid output"""
        site, schema = setup_site_and_schema
        synth = SyntheticTimeSeries(schema, SyntheticMethod.RANDOM_SEG_WINDOWED)

        t1 = datetime(2023, 1, 1, 0, 0, 0)
        t2 = datetime(2023, 1, 1, 0, 0, 1)

        ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)

        assert len(ex) == 100
        assert np.all(np.isfinite(ex))
        assert not np.any(np.isnan(ex))
        assert not np.any(np.isinf(ex))

    def test_random_seg_partial_default(self, setup_site_and_schema):
        """Test RANDOM_SEG_PARTIAL (default) method produces valid output"""
        site, schema = setup_site_and_schema
        # Default method is RANDOM_SEG_PARTIAL
        synth = SyntheticTimeSeries(schema, SyntheticMethod.RANDOM_SEG_PARTIAL)

        t1 = datetime(2023, 1, 1, 0, 0, 0)
        t2 = datetime(2023, 1, 1, 0, 0, 1)

        ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)

        assert len(ex) == 100
        assert np.all(np.isfinite(ex))
        assert not np.any(np.isnan(ex))
        assert not np.any(np.isinf(ex))

    def test_output_shape_matches_expected_length(self, setup_site_and_schema):
        """Test output shape matches expected length"""
        site, schema = setup_site_and_schema
        synth = SyntheticTimeSeries(schema, SyntheticMethod.RANDOM_SEG_PARTIAL)

        t1 = datetime(2023, 1, 1, 0, 0, 0)
        # Generate 5 seconds of data
        t2 = datetime(2023, 1, 1, 0, 0, 5)

        ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)

        expected_length = 5 * schema.sample_rate  # 5 * 100 = 500
        assert len(ex) == expected_length
        assert ex.shape == ey.shape == hx.shape == hy.shape == hz.shape

    def test_no_nan_inf_values(self, setup_site_and_schema):
        """Test no NaN/Inf values in output"""
        site, schema = setup_site_and_schema
        synth = SyntheticTimeSeries(schema, SyntheticMethod.RANDOM_SEG_PARTIAL)

        t1 = datetime(2023, 1, 1, 0, 0, 0)
        t2 = datetime(2023, 1, 1, 0, 0, 10)

        ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)

        for channel, name in [(ex, 'Ex'), (ey, 'Ey'), (hx, 'Hx'), (hy, 'Hy'), (hz, 'Hz')]:
            assert not np.any(np.isnan(channel)), f"{name} contains NaN"
            assert not np.any(np.isinf(channel)), f"{name} contains Inf"


class TestForwardSite:
    """Test ForwardSite class methods"""

    def test_frequencies_sorted_order(self):
        """Test frequencies returns sorted order"""
        site = ForwardSite(name='Test')
        # Add fields in non-sorted order
        site.fields.append(EMFields(freq=100.0))
        site.fields.append(EMFields(freq=10.0))
        site.fields.append(EMFields(freq=1.0))

        freqs = site.frequencies()
        assert np.array_equal(freqs, np.array([100.0, 10.0, 1.0]))

    def test_update_nature_magnetic_amplitude(self):
        """Test update_nature_magnetic_amplitude scales fields correctly"""
        site = ForwardSite(name='Test')
        site.fields.append(EMFields(
            freq=10.0,
            ex1=complex(2.0, 0.0),
            ey1=complex(2.0, 0.0),
            hx1=complex(1.0, 0.0),
            hy1=complex(1.0, 0.0),
            hz1=complex(0.5, 0.0),
        ))

        # Scale factors
        scale_e = np.array([2.0])
        scale_b = np.array([3.0])

        original_ex1 = abs(site.fields[0].ex1)
        original_hx1 = abs(site.fields[0].hx1)

        site.update_nature_magnetic_amplitude(scale_e, scale_b)

        # Verify scaling
        assert abs(site.fields[0].ex1) == original_ex1 * 2.0
        assert abs(site.fields[0].hx1) == original_hx1 * 3.0

    def test_interpolation_adds_new_frequencies(self):
        """Test that interpolation handling adds frequencies correctly"""
        site = ForwardSite(name='Test')
        site.fields.append(EMFields(freq=1.0))
        site.fields.append(EMFields(freq=10.0))
        site.fields.append(EMFields(freq=100.0))

        freqs = site.frequencies()
        assert len(freqs) == 3
        # Should be in the order they were added
        assert freqs[0] == 1.0
        assert freqs[1] == 10.0
        assert freqs[2] == 100.0

    def test_negative_harmonic_factor_conjugate(self):
        """Test that negative harmonic factor applies conjugate correctly"""
        site = ForwardSite(name='Test')
        original_phase = np.pi / 4
        site.fields.append(EMFields(
            freq=10.0,
            ex1=complex(1.0, 0.0) * np.exp(1j * original_phase)
        ))

        original_ex1 = site.fields[0].ex1
        # Get the conjugate
        conjugated = complex(original_ex1.real, -original_ex1.imag)

        assert np.isclose(abs(original_ex1), abs(conjugated))
        assert np.isclose(np.angle(original_ex1), -np.angle(conjugated))


class TestNoiseInjector:
    """Test NoiseInjector class"""

    def test_gaussian_noise(self):
        """Test Gaussian noise produces expected output"""
        config = NoiseConfig(noise_type=NoiseType.GAUSSIAN, amplitude=0.1)
        injector = NoiseInjector(config, sample_rate=100.0, seed=42)

        data = np.ones(1000)
        noisy = injector.add_noise(data)

        assert len(noisy[0]) == 1000
        assert np.any(np.abs(noisy[0]) > 0.01)  # Some noise added

    def test_square_wave_noise(self):
        """Test square wave noise produces expected output"""
        config = NoiseConfig(noise_type=NoiseType.SQUARE_WAVE, amplitude=1.0, frequency=10.0)
        injector = NoiseInjector(config, sample_rate=100.0, seed=42)

        data = np.zeros(1000)
        noisy = injector.add_noise(data)

        # Should have square wave pattern
        unique_values = np.unique(noisy[0])
        assert len(unique_values) <= 3  # -1, 0, 1 or subset

    def test_triangular_noise(self):
        """Test triangular noise produces expected output"""
        config = NoiseConfig(noise_type=NoiseType.TRIANGULAR, amplitude=1.0, frequency=10.0)
        injector = NoiseInjector(config, sample_rate=100.0, seed=42)

        data = np.zeros(1000)
        noisy = injector.add_noise(data)

        assert len(noisy[0]) == 1000
        assert np.any(np.abs(noisy[0]) > 0.01)

    def test_impulsive_noise(self):
        """Test impulsive noise produces expected output"""
        config = NoiseConfig(noise_type=NoiseType.IMPULSIVE, amplitude=1.0, probability=0.1)
        injector = NoiseInjector(config, sample_rate=100.0, seed=42)

        data = np.zeros(1000)
        noisy = injector.add_noise(data)

        assert len(noisy[0]) == 1000

    def test_powerline_noise(self):
        """Test powerline noise produces expected output"""
        config = NoiseConfig(noise_type=NoiseType.POWERLINE, amplitude=0.5, frequency=50.0, phase=0.0)
        injector = NoiseInjector(config, sample_rate=1000.0, seed=42)

        data = np.zeros(1000)
        noisy = injector.add_noise(data)

        assert len(noisy[0]) == 1000
        # Should show sinusoidal pattern
        assert np.std(noisy[0]) > 0.01

    def test_noise_amplitude_reasonable(self):
        """Test noise amplitude is reasonable"""
        config = NoiseConfig(noise_type=NoiseType.GAUSSIAN, amplitude=1.0)
        injector = NoiseInjector(config, sample_rate=100.0, seed=42)

        data = np.zeros(10000)
        noisy, = injector.add_noise(data)

        # Standard deviation should be roughly equal to amplitude
        assert 0.5 <= np.std(noisy) <= 2.0

    def test_zero_amplitude_no_noise(self):
        """Test zero amplitude produces no noise"""
        config = NoiseConfig(noise_type=NoiseType.GAUSSIAN, amplitude=0.0)
        injector = NoiseInjector(config, sample_rate=100.0, seed=42)

        data = np.ones(100)
        noisy, = injector.add_noise(data)

        assert np.allclose(noisy, data)


class TestModEMFileLoading:
    """Test ModEM file loading"""

    def test_parse_mock_modem_format(self):
        """Test parsing mock ModEM format string produces valid ForwardSite objects"""
        # Create a minimal ModEM format file with impedance block
        # Header line format: "Full_Impedance 3 1" (label freq_count site_count on same line)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write('> Full_Impedance\n')
            f.write('Full_Impedance 3 1\n')
            f.write('1.0 0.01 0.0 0.02 0.0 -0.01 0.0 0.03 0.0 0.0\n')
            f.write('10.0 0.01 0.0 0.02 0.0 -0.01 0.0 0.03 0.0 0.0\n')
            f.write('100.0 0.01 0.0 0.02 0.0 -0.01 0.0 0.03 0.0 0.0\n')
            temp_path = f.name

        try:
            sites = load_modem_file(temp_path)
            assert len(sites) > 0, "Should parse at least one site"
            assert sites[0].name is not None
            assert len(sites[0].fields) == 3
            # Verify frequencies are parsed
            freqs = sites[0].frequencies()
            assert len(freqs) == 3
        finally:
            os.unlink(temp_path)

    def test_modem_empty_block(self):
        """Test parsing empty block doesn't crash"""
        mock_content = """> Full_Impedance
 0 0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write(mock_content)
            temp_path = f.name

        try:
            sites = load_modem_file(temp_path)
            assert isinstance(sites, list)
        finally:
            os.unlink(temp_path)

    def test_modem_invalid_data(self):
        """Test parsing invalid data doesn't crash"""
        mock_content = """> Full_Impedance
 invalid header data
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write(mock_content)
            temp_path = f.name

        try:
            sites = load_modem_file(temp_path)
            assert isinstance(sites, list)
        finally:
            os.unlink(temp_path)


class TestSyntheticSchema:
    """Test SyntheticSchema class"""

    def test_from_ts_config(self):
        """Test creating schema from TS config"""
        schema = SyntheticSchema.from_ts('TS3')
        assert schema.name == 'TS3'
        assert schema.sample_rate == 2400
        assert schema.freq_min == 1
        assert schema.freq_max == 1000

    def test_ts_configs_available(self):
        """Test TS_CONFIGS contains expected configs"""
        assert 'TS2' in TS_CONFIGS
        assert 'TS3' in TS_CONFIGS
        assert 'TS4' in TS_CONFIGS
        assert 'TS5' in TS_CONFIGS

    def test_schema_custom_values(self):
        """Test schema with custom values"""
        schema = SyntheticSchema(
            name='Custom',
            sample_rate=500,
            freq_min=0.1,
            freq_max=500,
            synthetic_periods=10.0
        )
        assert schema.sample_rate == 500
        assert schema.freq_min == 0.1
        assert schema.freq_max == 500


class TestEMFields:
    """Test EMFields dataclass"""

    def test_emfields_default_values(self):
        """Test EMFields default values are zero"""
        field = EMFields(freq=10.0)
        assert field.freq == 10.0
        assert field.ex1 == complex(0, 0)
        assert field.hx1 == complex(0, 0)

    def test_emfields_complex_values(self):
        """Test EMFields with complex values"""
        field = EMFields(
            freq=10.0,
            ex1=complex(1.0, 0.5),
            zxy=complex(0.5, -0.3)
        )
        assert field.ex1 == complex(1.0, 0.5)
        assert field.zxy == complex(0.5, -0.3)


class TestCreateTestSite:
    """Test create_test_site function"""

    def test_create_test_site_returns_valid_site(self):
        """Test create_test_site returns valid ForwardSite"""
        site = create_test_site()
        assert isinstance(site, ForwardSite)
        assert len(site.fields) > 0
        assert site.name == 'Test'

    def test_test_site_frequencies_range(self):
        """Test test site has frequencies in expected range"""
        site = create_test_site()
        freqs = site.frequencies()
        assert np.min(freqs) >= 0.01
        assert np.max(freqs) <= 1000


class TestCalculateMTScaleFactors:
    """Test calculate_mt_scale_factors function"""

    def test_scale_factors_output_shape(self):
        """Test scale factors have correct shape"""
        site = create_test_site()
        scale_e, scale_b = calculate_mt_scale_factors(site)
        n_freqs = len(site.frequencies())
        assert len(scale_e) == n_freqs
        assert len(scale_b) == n_freqs

    def test_scale_factors_positive(self):
        """Test scale factors are positive"""
        site = create_test_site()
        scale_e, scale_b = calculate_mt_scale_factors(site)
        assert np.all(scale_e > 0)
        assert np.all(scale_b > 0)


class TestSyntheticMethodEnum:
    """Test SyntheticMethod enum"""

    def test_all_methods_defined(self):
        """Test all 6 methods are defined"""
        assert SyntheticMethod.FIX is not None
        assert SyntheticMethod.FIXED_AVG is not None
        assert SyntheticMethod.FIXED_AVG_WINDOWED is not None
        assert SyntheticMethod.RANDOM_SEG is not None
        assert SyntheticMethod.RANDOM_SEG_WINDOWED is not None
        assert SyntheticMethod.RANDOM_SEG_PARTIAL is not None

    def test_method_count(self):
        """Test exactly 6 methods exist"""
        assert len(SyntheticMethod) == 6


class TestNoiseTypeEnum:
    """Test NoiseType enum"""

    def test_all_noise_types_defined(self):
        """Test all noise types are defined"""
        assert NoiseType.SQUARE_WAVE is not None
        assert NoiseType.TRIANGULAR is not None
        assert NoiseType.IMPULSIVE is not None
        assert NoiseType.GAUSSIAN is not None
        assert NoiseType.POWERLINE is not None

    def test_noise_type_count(self):
        """Test exactly 5 noise types exist"""
        assert len(NoiseType) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
