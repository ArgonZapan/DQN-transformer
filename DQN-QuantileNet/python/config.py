import os
import tomllib
import torch

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.toml')

REQUIRED_SECTIONS = ['actors', 'zmq', 'timeframes', 'prediction', 'model',
                     'training', 'features', 'data', 'logging', 'tensorboard']


def load_config() -> dict:
    with open(CONFIG_PATH, 'rb') as f:
        cfg = tomllib.load(f)

    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            raise ValueError(f'config.toml: missing section [{section}]')

    if not cfg.get('actors'):
        raise ValueError('config.toml: at least one [[actors]] entry required')

    active_tfs = [tf for tf in ['candles_1m', 'candles_15m', 'candles_1h',
                                 'candles_4h', 'candles_1d', 'candles_1w']
                  if cfg['timeframes'].get(tf, 0) > 0]
    if not active_tfs:
        raise ValueError('config.toml: at least one timeframe must be > 0')

    mode = cfg['prediction'].get('mode', 'both')
    if mode in ('quantile', 'both') and not cfg['prediction'].get('quantiles'):
        raise ValueError('config.toml: prediction.quantiles required for mode=quantile/both')
    if mode in ('threshold', 'both') and not cfg['prediction'].get('thresholds_pct'):
        raise ValueError('config.toml: prediction.thresholds_pct required for mode=threshold/both')

    return cfg


def get_device(config: dict) -> torch.device:
    requested = config['training'].get('device', 'auto')
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        import logging
        logging.getLogger(__name__).warning('CUDA requested but not available — falling back to CPU.')
        return torch.device('cpu')
    return torch.device(requested)
