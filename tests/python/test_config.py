import os
import sys
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from config import (
    load_config, _validate_config,
    get_learner_config, get_training_config, get_model_config,
    get_features_config, get_reward_config, get_per_config,
    get_timeframes_config, get_monitoring_config, get_data_config,
    get_logging_config, get_backtesting_config, get_actors_config
)


@pytest.fixture
def config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.toml')
    return load_config(config_path)


class TestLoadConfig:
    def test_loads_valid_config(self, config):
        assert config is not None
        assert isinstance(config, dict)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.toml')

    def test_raises_on_missing_section(self):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                tmp_path = f.name
                f.write('[learner]\nhost = "tcp://127.0.0.1"\nport = 5555\n')
            with pytest.raises(ValueError, match="Missing required config section"):
                load_config(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_raises_on_no_actors(self):
        minimal = {
            'learner': {'host': 'tcp://127.0.0.1', 'port': 5555, 'metrics_port': 5556, 'device': 'cuda'},
            'training': {
                'gamma': 0.999, 'lr': 0.0001, 'batch_size': 256,
                'buffer_capacity': 500000, 'min_buffer_size': 10000,
                'target_update_interval': 1000, 'epsilon_start': 1.0,
                'epsilon_end': 0.05, 'epsilon_decay_fraction': 0.3,
                'dropout': 0.1, 'n_step': 1, 'seed': 42,
                'checkpoint_interval': 5000, 'evaluation_interval': 10000,
                'train_data_fraction': 0.8, 'min_episode_length': 100
            },
            'model': {
                'num_actions': 4, 'n_transformer_blocks': 3,
                'n_attention_heads': 8, 'key_dim': 64, 'ff_dim': 512,
                'conv_kernel_size': 3, 'conv1d_filters': 128
            },
            'features': {'num_features': 8},
            'reward': {'commission_open': 0.001},
            'data': {'source': 'file'},
            'logging': {'level': 'INFO'},
            'timeframes': {'candles_1m': 15},
            'monitoring': {'host': 'tcp://127.0.0.1'},
            'per': {'alpha': 0.6},
            'backtesting': {'enabled': False},
            'api': {'simulation_mode': True},
        }
        with pytest.raises(ValueError, match="at least one"):
            _validate_config(minimal)


class TestConfigSections:
    def test_learner_config(self, config):
        learner = get_learner_config(config)
        assert learner['host'] == 'tcp://127.0.0.1'
        assert learner['port'] == 5555
        assert learner['metrics_port'] == 5556
        assert learner['device'] == 'cuda'

    def test_training_config(self, config):
        training = get_training_config(config)
        assert training['gamma'] == 0.999
        assert training['lr'] == 0.0001
        assert training['batch_size'] == 256
        assert training['buffer_capacity'] == 500000
        assert training['epsilon_start'] == 1.0
        assert training['epsilon_end'] == 0.05
        assert training['seed'] == 42

    def test_model_config(self, config):
        model = get_model_config(config)
        assert model['num_actions'] == 4
        assert model['n_transformer_blocks'] == 3
        assert model['n_attention_heads'] == 8
        assert model['key_dim'] == 64
        assert model['ff_dim'] == 512
        assert model['conv_kernel_size'] == 3
        assert model['conv1d_filters'] == 128

    def test_features_config(self, config):
        features = get_features_config(config)
        assert features['num_features'] == 8
        assert features['use_ohlcv'] is True
        assert features['use_rsi'] is True

    def test_per_config(self, config):
        per = get_per_config(config)
        assert per['alpha'] == 0.6
        assert per['beta_start'] == 0.4
        assert per['beta_end'] == 1.0

    def test_reward_config(self, config):
        reward = get_reward_config(config)
        assert reward['commission_open'] == 0.001
        assert reward['clip_min'] == -1.0
        assert reward['clip_max'] == 1.0

    def test_timeframes_config(self, config):
        tf = get_timeframes_config(config)
        assert tf['candles_1m'] == 15
        assert tf['candles_15m'] == 15
        assert tf['candles_1h'] == 20
        assert tf['candles_1d'] == 30
        assert tf['candles_1w'] == 54

    def test_monitoring_config(self, config):
        mon = get_monitoring_config(config)
        assert mon['port'] == 3001
        assert mon['metrics_pull_port'] == 3002

    def test_data_config(self, config):
        data = get_data_config(config)
        assert data['source'] == 'file'
        assert data['normalization_window'] == 20

    def test_actors_config(self, config):
        actors = get_actors_config(config)
        assert len(actors) == 3
        symbols = [a['symbol'] for a in actors]
        assert 'BTCUSDT' in symbols
        assert 'ETHUSDT' in symbols
        assert 'SOLUSDT' in symbols

    def test_backtesting_config(self, config):
        bt = get_backtesting_config(config)
        assert bt['enabled'] is False

    def test_logging_config(self, config):
        log = get_logging_config(config)
        assert log['level'] == 'INFO'
