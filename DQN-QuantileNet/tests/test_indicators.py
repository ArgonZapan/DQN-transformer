"""Feature-causality tests for the QuantileNet indicator stack.

`build_all_windows_vectorized` computes features ONCE on the full per-TF series
and slices per candle, while `build_window_at` recomputes features on data cut
at each historical anchor. The two are equivalent — and the vectorized path is
free of look-ahead — only if every feature is causal: feature[i] must depend on
data[:i+1] alone. These tests lock that invariant so a future non-causal feature
(e.g. a full-series z-score) can't silently leak the future into past windows.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from data.indicators import (  # noqa: E402
    compute_features, rolling_mean, rolling_std, rsi, ema, sma, stochastic_k,
)


def _ohlcv(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    high  = close + np.abs(rng.standard_normal(n))
    low   = close - np.abs(rng.standard_normal(n))
    open_ = close + rng.standard_normal(n)
    vol   = np.abs(rng.standard_normal(n)) * 1000.0 + 10.0
    return open_, high, low, close, vol


class TestFeatureCausality:
    def test_compute_features_matches_prefix(self):
        o, h, l, c, v = _ohlcv(300)
        norm = 60
        full = compute_features(o, h, l, c, v, norm)
        for j in (70, 120, 200, 299):
            partial = compute_features(o[:j + 1], h[:j + 1], l[:j + 1], c[:j + 1], v[:j + 1], norm)
            assert partial.shape[0] == j + 1
            # Row j of the full computation must equal row j computed from data[:j+1].
            np.testing.assert_allclose(
                partial[j], full[j], rtol=1e-5, atol=1e-6,
                err_msg=f'feature row {j} depends on future data (look-ahead!)',
            )


class TestIndicatorCausality:
    def setup_method(self):
        _, h, l, c, _ = _ohlcv(200, seed=7)
        self.h, self.l, self.c = h, l, c

    def _assert_causal(self, fn, *arrs, positions=(40, 90, 150, 199)):
        full = fn(*arrs)
        for j in positions:
            partial = fn(*[a[:j + 1] for a in arrs])
            np.testing.assert_allclose(
                partial[j], full[j], rtol=1e-6, atol=1e-8,
                err_msg=f'{fn.__name__}[{j}] changed when future was trimmed',
            )

    def test_rolling_mean_causal(self):
        self._assert_causal(lambda a: rolling_mean(a, 60), self.c)

    def test_rolling_std_causal(self):
        self._assert_causal(lambda a: rolling_std(a, 60), self.c)

    def test_sma_causal(self):
        self._assert_causal(lambda a: sma(a, 20), self.c)

    def test_ema_causal(self):
        self._assert_causal(lambda a: ema(a, 12), self.c)

    def test_rsi_causal(self):
        self._assert_causal(lambda a: rsi(a, 14), self.c)

    def test_stochastic_k_causal(self):
        self._assert_causal(lambda h, l, c: stochastic_k(h, l, c, 14), self.h, self.l, self.c)
