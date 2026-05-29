"""Tests for python/backtest.py pure helpers and state assembly.

Guards the feature-width fix (build_state_fast must front-pad to the configured
num_features, not a hardcoded 8) and documents the fee/mask conventions.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import backtest  # noqa: E402
from backtest import action_mask_for, net_pnl, build_state_fast, Position  # noqa: E402


class TestActionMask:
    def test_flat_allows_open_not_close(self):
        assert action_mask_for(None) == [1, 1, 1, 0]

    def test_in_position_allows_hold_close(self):
        assert action_mask_for(Position('LONG', 100.0, 0)) == [0, 0, 1, 1]


class TestNetPnl:
    cfg = {'backtesting': {'fee': 0.00075}}

    def test_long_profit_minus_two_fees(self):
        p = Position('LONG', 100.0, 0)
        p.close(110.0, 1)
        expected = 0.1 - 0.00075 * 2
        assert net_pnl(p, self.cfg) == pytest.approx(expected)

    def test_short_profit_minus_two_fees(self):
        p = Position('SHORT', 100.0, 0)
        p.close(90.0, 1)
        expected = 0.1 - 0.00075 * 2
        assert net_pnl(p, self.cfg) == pytest.approx(expected)

    def test_losing_trade_is_negative(self):
        p = Position('LONG', 100.0, 0)
        p.close(99.0, 1)
        assert net_pnl(p, self.cfg) < 0


class TestBuildStateFast:
    @staticmethod
    def _cfg(num_features=11):
        return {
            'timeframes': {'candles_1m': 10, 'candles_15m': 0, 'candles_1h': 0,
                           'candles_1d': 0, 'candles_1w': 0},
            'features': {'num_features': num_features},
        }

    def test_pad_path_matches_feature_width(self):
        # Fewer candles than the window → front-padding must match the real width (11), not 8.
        cfg = self._cfg()
        feats = np.random.randn(5, 11).astype(np.float32)
        precomp = {'features': {'1m': feats}, 'alignment': {}}
        state = build_state_fast(precomp, 2, cfg)  # end=3, start=0, actual=(3,11), pad=7
        assert state['1m'].shape == (10, 11)
        assert np.allclose(state['1m'][:7], 0.0)              # zero front-pad
        assert np.allclose(state['1m'][7:], feats[0:3])       # real tail

    def test_full_window_no_pad(self):
        cfg = self._cfg()
        feats = np.random.randn(50, 11).astype(np.float32)
        precomp = {'features': {'1m': feats}, 'alignment': {}}
        state = build_state_fast(precomp, 20, cfg)  # end=21, start=11 → exactly 10 rows
        assert state['1m'].shape == (10, 11)
        assert np.allclose(state['1m'], feats[11:21])

    def test_higher_tf_uses_alignment(self):
        cfg = {
            'timeframes': {'candles_1m': 4, 'candles_15m': 3, 'candles_1h': 0,
                           'candles_1d': 0, 'candles_1w': 0},
            'features': {'num_features': 11},
        }
        feats_1m = np.random.randn(20, 11).astype(np.float32)
        feats_15m = np.random.randn(8, 11).astype(np.float32)
        # At 1m index 10, three 15m candles are closed.
        alignment = np.zeros(20, dtype=np.int32)
        alignment[10] = 3
        precomp = {'features': {'1m': feats_1m, '15m': feats_15m},
                   'alignment': {'15m': alignment}}
        state = build_state_fast(precomp, 10, cfg)
        assert state['1m'].shape == (4, 11)
        assert state['15m'].shape == (3, 11)
        assert np.allclose(state['15m'], feats_15m[0:3])
