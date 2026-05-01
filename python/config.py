import os
import toml
import torch


def get_optimal_device():
    """Automatically selects the best available device (GPU or CPU)."""
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
    config = toml.load(config_path)

    _validate_config(config)

    # Handle automatic device selection
    device = config['learner'].get('device', 'auto')
    if device == 'auto':
        config['learner']['device'] = get_optimal_device()
    elif device == 'cuda' and not torch.cuda.is_available():
        print(f"[Config] WARNING: device='cuda' requested but GPU not available. Falling back to CPU.")
        config['learner']['device'] = 'cpu'

    return config


def _require_keys(d, keys, label):
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing required {label} config: {', '.join(missing)}")


def _validate_config(config):
    _require_keys(config, [
        'learner', 'training', 'model', 'features',
        'reward', 'data', 'logging', 'timeframes',
        'monitoring', 'per', 'backtesting', 'api',
    ], 'section')

    if 'actors' not in config or len(config['actors']) == 0:
        raise ValueError("Config must have at least one [[actors]] entry")

    _require_keys(config['learner'], ['host', 'port', 'metrics_port', 'device'], 'learner')
    _require_keys(config['training'], [
        'gamma', 'lr', 'batch_size', 'buffer_capacity', 'min_buffer_size',
        'target_update_interval', 'epsilon_start', 'epsilon_end',
        'epsilon_decay_fraction', 'dropout', 'seed',
        'checkpoint_interval', 'evaluation_interval',
        'validation_days', 'episode_length', 'step_interval', 'max_trades_per_episode',
    ], 'training')
    _require_keys(config['model'], [
        'num_actions', 'n_transformer_blocks', 'n_attention_heads',
        'ff_dim', 'conv_kernel_size', 'conv1d_filters',
    ], 'model')


