import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from config import load_config
from model.network import TradingDQN, Conv1DBlock, TransformerEncoderBlock
from model.noisy_linear import NoisyLinear


@pytest.fixture
def config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.toml')
    return load_config(config_path)


@pytest.fixture
def model(config):
    return TradingDQN(config)


def make_dummy_states(config, batch_size=4):
    tf_cfg = config['timeframes']
    num_features = config['features']['num_features']
    states = {}
    for key in sorted(tf_cfg.keys()):
        seq_len = tf_cfg[key]
        states[key] = torch.randn(batch_size, seq_len, num_features)
    return states


class TestConv1DBlock:
    def test_output_shape(self):
        block = Conv1DBlock(num_features=8, conv_filters=128, kernel_size=3, dropout=0.1)
        x = torch.randn(4, 15, 8)
        out = block(x)
        assert out.shape == (4, 128)

    def test_different_seq_lengths(self):
        block = Conv1DBlock(num_features=8, conv_filters=128, kernel_size=3, dropout=0.1)
        for seq_len in [15, 20, 30, 54]:
            x = torch.randn(2, seq_len, 8)
            out = block(x)
            assert out.shape == (2, 128)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerEncoderBlock(d_model=128, n_heads=8, ff_dim=512, dropout=0.1)
        x = torch.randn(4, 5, 128)
        out = block(x)
        assert out.shape == (4, 5, 128)

    def test_residual_connection(self):
        block = TransformerEncoderBlock(d_model=128, n_heads=8, ff_dim=512, dropout=0.0)
        block.eval()
        x = torch.randn(1, 5, 128)
        out = block(x)
        assert out.shape == x.shape


class TestTradingDQN:
    def test_output_shape(self, config, model):
        model.eval()
        batch_size = 4
        states = make_dummy_states(config, batch_size)
        q_values = model(states)
        assert q_values.shape == (batch_size, config['model']['num_actions'])

    def test_output_shape_list_input(self, config, model):
        model.eval()
        batch_size = 2
        tf_cfg = config['timeframes']
        num_features = config['features']['num_features']
        states = []
        for key in sorted(tf_cfg.keys()):
            seq_len = tf_cfg[key]
            states.append(torch.randn(batch_size, seq_len, num_features))
        q_values = model(states)
        assert q_values.shape == (batch_size, config['model']['num_actions'])

    def test_action_mask(self, config, model):
        model.eval()
        batch_size = 2
        states = make_dummy_states(config, batch_size)
        mask = torch.tensor([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.float32)
        q_values = model(states, action_mask=mask)
        assert q_values[0, 3] == float('-inf')
        assert q_values[1, 0] == float('-inf')
        assert q_values[1, 1] == float('-inf')

    def test_confidence_score(self, config, model):
        model.eval()
        states = make_dummy_states(config, batch_size=4)
        q_values = model(states)
        confidence = model.get_confidence(q_values)
        assert confidence.shape == (4,)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_no_transformer(self, config):
        config_copy = {k: dict(v) if isinstance(v, dict) else v for k, v in config.items()}
        config_copy['model'] = dict(config['model'])
        config_copy['model']['n_transformer_blocks'] = 0
        model = TradingDQN(config_copy)
        model.eval()
        states = make_dummy_states(config_copy, batch_size=2)
        q_values = model(states)
        assert q_values.shape == (2, config_copy['model']['num_actions'])

    def test_gradient_flow(self, config, model):
        states = make_dummy_states(config, batch_size=4)
        q_values = model(states)
        loss = q_values.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestNoisyLinear:
    def test_output_shape(self):
        layer = NoisyLinear(64, 32)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_noise_changes_output(self):
        layer = NoisyLinear(64, 32)
        layer.train()
        x = torch.randn(1, 64)
        out1 = layer(x).detach()
        layer.reset_noise()
        out2 = layer(x).detach()
        assert not torch.allclose(out1, out2)

    def test_eval_mode_no_noise(self):
        layer = NoisyLinear(64, 32)
        layer.eval()
        x = torch.randn(1, 64)
        out1 = layer(x).detach()
        layer.reset_noise()
        out2 = layer(x).detach()
        assert torch.allclose(out1, out2)
