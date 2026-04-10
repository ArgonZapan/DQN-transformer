import os
import toml
import torch


def get_optimal_device():
    """Automatycznie wybiera najlepsze dostępne urządzenie (GPU lub CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[Config] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[Config] CUDA version: {torch.version.cuda}")
    else:
        device = "cpu"
        print("[Config] No GPU available, using CPU")
    return device


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.toml')

    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = toml.load(config_path)

    _validate_config(config)

    # Obsługa automatycznego wyboru urządzenia
    device = config['learner'].get('device', 'auto')
    if device == 'auto':
        config['learner']['device'] = get_optimal_device()
    elif device == 'cuda' and not torch.cuda.is_available():
        print(f"[Config] WARNING: device='cuda' requested but GPU not available. Falling back to CPU.")
        config['learner']['device'] = 'cpu'

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
        'epsilon_decay_fraction', 'dropout', 'seed',
        'checkpoint_interval', 'evaluation_interval',
        'train_data_fraction', 'episode_length', 'step_interval', 'max_trades_per_episode'
    ]
    for key in required_training:
        if key not in config['training']:
            raise ValueError(f"Missing required training config: {key}")

    required_model = [
        'num_actions', 'n_transformer_blocks', 'n_attention_heads',
        'ff_dim', 'conv_kernel_size', 'conv1d_filters'
    ]
    for key in required_model:
        if key not in config['model']:
            raise ValueError(f"Missing required model config: {key}")


