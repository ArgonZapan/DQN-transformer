"""Config tests aligned with the current direct-dict config API.

The historical ``get_*_config`` accessor helpers were removed in favour of plain
dict access (``config['learner']`` etc.). These tests validate ``load_config``
and ``_validate_config`` against the real ``config.toml`` and the current schema.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from config import load_config, _validate_config  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config.toml')

REQUIRED_SECTIONS = [
    'learner', 'training', 'model', 'features',
    'reward', 'data', 'logging', 'timeframes',
    'monitoring', 'per', 'backtesting', 'api',
]


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


class TestLoadConfig:
    def test_loads_valid_config(self, config):
        assert isinstance(config, dict)
        for section in REQUIRED_SECTIONS:
            assert section in config, f"missing section {section}"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.toml')

    def test_device_resolved(self, config):
        # 'auto'/'cuda' are resolved to a concrete device by load_config.
        assert config['learner']['device'] in ('cuda', 'cpu')


class TestValidation:
    def test_raises_on_missing_section(self):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                tmp_path = f.name
                f.write('[learner]\nhost = "tcp://127.0.0.1"\nport = 5555\n')
            with pytest.raises(ValueError, match="Missing required"):
                load_config(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_raises_on_no_actors(self):
        minimal = {s: {} for s in REQUIRED_SECTIONS}
        with pytest.raises(ValueError, match="at least one"):
            _validate_config(minimal)


class TestSchemaValues:
    def test_actors(self, config):
        symbols = [a['symbol'] for a in config['actors']]
        assert len(symbols) >= 1
        assert 'BTCUSDT' in symbols

    def test_training_invariants(self, config):
        t = config['training']
        assert 0.0 < t['gamma'] <= 1.0
        assert t['lr'] > 0
        assert t['batch_size'] > 0
        assert t['buffer_capacity'] >= t['min_buffer_size']
        assert t['epsilon_start'] >= t['epsilon_end']

    def test_model_features_consistency(self, config):
        assert config['model']['num_actions'] == 4
        assert config['features']['num_features'] > 0
        # d_model (= conv1d_filters) must be divisible by attention heads.
        assert config['model']['conv1d_filters'] % config['model']['n_attention_heads'] == 0

    def test_per_bounds(self, config):
        per = config['per']
        assert 0.0 <= per['alpha'] <= 1.0
        assert 0.0 <= per['beta_start'] <= per['beta_end'] <= 1.0

    def test_timeframes_non_negative(self, config):
        for k, v in config['timeframes'].items():
            assert v >= 0, f"{k} must be >= 0"
        assert any(v > 0 for v in config['timeframes'].values())
