"""Model tests for the current TradingDQN architecture.

Architecture notes that shaped these tests:
- Conv1DBlock keeps the full causal sequence ([B, T, filters]); GAP happens
  later, after the Transformer.
- The position branch is only exercised when ``position_features`` is supplied;
  otherwise a zero vector bypasses ``pos_fc`` (so it receives no gradient).
- The position vector is ``model.position_dim`` wide (10 base + quantile slots).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from config import load_config  # noqa: E402
from model.network import TradingDQN, Conv1DBlock, TransformerEncoderBlock  # noqa: E402
from model.noisy_linear import NoisyLinear  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config.toml')


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def model(config):
    return TradingDQN(config)


def enabled_timeframes(config):
    return [k for k, v in config['timeframes'].items() if v > 0]


def make_dummy_states(config, batch_size=4):
    num_features = config['features']['num_features']
    states = {}
    for key in enabled_timeframes(config):
        states[key] = torch.randn(batch_size, config['timeframes'][key], num_features)
    return states


class TestConv1DBlock:
    def test_keeps_sequence(self):
        block = Conv1DBlock(num_features=8, conv_filters=64, kernel_size=3, dropout=0.1)
        x = torch.randn(4, 15, 8)
        out = block(x)
        assert out.shape == (4, 15, 64)  # causal conv preserves length, no GAP

    def test_different_seq_lengths(self):
        block = Conv1DBlock(num_features=8, conv_filters=64, kernel_size=3, dropout=0.1)
        for seq_len in [14, 32, 48, 60]:
            out = block(torch.randn(2, seq_len, 8))
            assert out.shape == (2, seq_len, 64)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerEncoderBlock(d_model=64, n_heads=4, ff_dim=128, dropout=0.1)
        x = torch.randn(4, 7, 64)
        assert block(x).shape == (4, 7, 64)


class TestTradingDQN:
    def test_output_shape(self, config, model):
        model.eval()
        q = model(make_dummy_states(config, 4))
        assert q.shape == (4, config['model']['num_actions'])

    def test_output_shape_list_input(self, config, model):
        model.eval()
        num_features = config['features']['num_features']
        states = [torch.randn(2, config['timeframes'][k], num_features)
                  for k in enabled_timeframes(config)]
        q = model(states)
        assert q.shape == (2, config['model']['num_actions'])

    def test_action_mask_blocks_actions(self, config, model):
        model.eval()
        states = make_dummy_states(config, 2)
        mask = torch.tensor([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.float32)
        q = model(states, action_mask=mask)
        assert q[0, 3] == float('-inf')
        assert q[1, 0] == float('-inf')
        assert q[1, 1] == float('-inf')

    def test_position_features_shape(self, config, model):
        model.eval()
        states = make_dummy_states(config, 3)
        pos = torch.randn(3, model.position_dim)
        q = model(states, position_features=pos)
        assert q.shape == (3, config['model']['num_actions'])

    def test_no_transformer(self, config):
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
        cfg['model'] = dict(config['model'])
        cfg['model']['n_transformer_blocks'] = 0
        m = TradingDQN(cfg)
        m.eval()
        q = m(make_dummy_states(cfg, 2))
        assert q.shape == (2, cfg['model']['num_actions'])

    def test_gradient_flow(self, config, model):
        model.train()
        states = make_dummy_states(config, 4)
        pos = torch.randn(4, model.position_dim)  # exercise the pos_fc branch too
        loss = model(states, position_features=pos).sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestNoisyLinear:
    def test_output_shape(self):
        layer = NoisyLinear(64, 32)
        assert layer(torch.randn(4, 64)).shape == (4, 32)

    def test_noise_changes_output_in_train(self):
        layer = NoisyLinear(64, 32)
        layer.train()
        x = torch.randn(1, 64)
        out1 = layer(x).detach()
        layer.reset_noise()
        out2 = layer(x).detach()
        assert not torch.allclose(out1, out2)

    def test_eval_mode_is_deterministic(self):
        layer = NoisyLinear(64, 32)
        layer.eval()
        x = torch.randn(1, 64)
        out1 = layer(x).detach()
        layer.reset_noise()
        out2 = layer(x).detach()
        assert torch.allclose(out1, out2)
