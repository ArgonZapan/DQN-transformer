import os
import toml


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.toml')

    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = toml.load(config_path)

    _validate_config(config)

    return config


def _validate_config(config):
    required_sections = [
        'learner', 'training', 'model', 'features',
        'reward', 'data', 'logging', 'timeframes',
        'monitoring', 'per', 'backtesting', 'api'
    ]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: [{section}]")

    if 'actors' not in config or len(config['actors']) == 0:
        raise ValueError("Config must have at least one [[actors]] entry")

    required_learner = ['host', 'port', 'metrics_port', 'device']
    for key in required_learner:
        if key not in config['learner']:
            raise ValueError(f"Missing required learner config: {key}")

    required_training = [
        'gamma', 'lr', 'batch_size', 'buffer_capacity', 'min_buffer_size',
        'target_update_interval', 'epsilon_start', 'epsilon_end',
        'epsilon_decay_fraction', 'dropout', 'n_step', 'seed',
        'checkpoint_interval', 'evaluation_interval',
        'train_data_fraction', 'min_episode_length'
    ]
    for key in required_training:
        if key not in config['training']:
            raise ValueError(f"Missing required training config: {key}")

    required_model = [
        'num_actions', 'n_transformer_blocks', 'n_attention_heads',
        'key_dim', 'ff_dim', 'conv_kernel_size', 'conv1d_filters'
    ]
    for key in required_model:
        if key not in config['model']:
            raise ValueError(f"Missing required model config: {key}")


def get_learner_config(config):
    return config['learner']


def get_training_config(config):
    return config['training']


def get_model_config(config):
    return config['model']


def get_features_config(config):
    return config['features']


def get_reward_config(config):
    return config['reward']


def get_per_config(config):
    return config['per']


def get_timeframes_config(config):
    return config['timeframes']


def get_monitoring_config(config):
    return config['monitoring']


def get_data_config(config):
    return config['data']


def get_logging_config(config):
    return config['logging']


def get_backtesting_config(config):
    return config['backtesting']


def get_actors_config(config):
    return config['actors']
