"""Regression tests for PrioritizedReplayBuffer priority bookkeeping.

These guard the invariant that tree leaves and ``max_priority`` both store the
*final* priority p = (|td| + eps)^alpha, and that fresh experiences enter the
tree at exactly the current ``max_priority`` (standard PER) — not at
``max_priority ** alpha``, which previously under-weighted new transitions
whenever ``max_priority`` grew above 1.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from training.prioritized_buffer import PrioritizedReplayBuffer  # noqa: E402


def make_config(capacity=64, alpha=0.6):
    return {
        'training': {'buffer_capacity': capacity},
        'features': {'num_features': 4},
        'model': {'num_actions': 4},
        'learner': {'device': 'cpu'},
        'per': {'alpha': alpha, 'beta_start': 0.4, 'beta_end': 1.0, 'epsilon': 1e-4},
        'timeframes': {'candles_1m': 5},
        'quantilenet': {'enabled': False},
    }


def _leaf(buffer, data_idx):
    """Read the tree leaf priority for a given data index."""
    return buffer.tree.tree[data_idx + buffer.tree.capacity - 1]


def _exp(action=2, reward=0.0):
    """Minimal experience tuple (state, action, reward, next_state, done, mask, next_mask)."""
    return ({}, action, reward, {}, False, None, None)


def test_fresh_experience_enters_at_max_priority():
    cfg = make_config()
    buf = PrioritizedReplayBuffer(cfg)
    alpha = cfg['per']['alpha']

    # Seed a few experiences, then assign a large TD so max_priority climbs well above 1.
    buf.batch_add([_exp() for _ in range(4)])
    big_td = 10.0
    buf.update_priorities(np.arange(4), np.full(4, big_td))

    expected_max = (big_td + cfg['per']['epsilon']) ** alpha
    assert buf.max_priority == pytest.approx(expected_max, rel=1e-6)
    assert expected_max > 1.0  # precondition that exposes the old double-alpha bug

    # A brand-new experience must enter at max_priority, NOT max_priority**alpha.
    pos_before = buf.position
    buf.batch_add([_exp()])
    leaf = _leaf(buf, pos_before)
    assert leaf == pytest.approx(buf.max_priority, rel=1e-6)
    # Guard against the regression explicitly.
    assert leaf != pytest.approx(buf.max_priority ** alpha, rel=1e-6)


def test_add_path_matches_max_priority():
    cfg = make_config()
    buf = PrioritizedReplayBuffer(cfg)

    buf.batch_add([_exp() for _ in range(4)])
    buf.update_priorities(np.arange(4), np.full(4, 8.0))

    pos_before = buf.position
    buf.add({}, 2, 0.0, {}, False, None)  # no td_error → should use max_priority
    assert _leaf(buf, pos_before) == pytest.approx(buf.max_priority, rel=1e-6)


def test_non_finite_td_does_not_poison_tree():
    cfg = make_config()
    buf = PrioritizedReplayBuffer(cfg)
    buf.batch_add([_exp() for _ in range(4)])

    td = np.array([np.inf, np.nan, 2.0, 3.0])
    buf.update_priorities(np.arange(4), td)

    assert np.isfinite(buf.tree.total())
    assert np.isfinite(buf.max_priority)
    # Sampling must still succeed with a healthy tree.
    out = buf.sample(2)
    assert len(out) == 11


def test_priorities_are_alpha_applied_and_sampleable():
    cfg = make_config()
    buf = PrioritizedReplayBuffer(cfg)
    buf.batch_add([_exp(reward=float(i)) for i in range(8)])
    buf.update_priorities(np.arange(8), np.linspace(0.5, 5.0, 8))

    # Higher TD must map to higher tree priority (monotonic under alpha>0).
    leaves = [_leaf(buf, i) for i in range(8)]
    assert leaves == sorted(leaves)


def test_update_batch_duplicate_indices_keep_tree_consistent():
    """Duplicated sample indices must not drift ancestor sums.

    The leaf assignment in ``update_batch`` is last-wins, but the upward
    propagation used ``np.add.at`` which accumulated EVERY duplicate's delta —
    after enough updates the root total diverged from the true sum of leaves,
    distorting sampling probabilities and IS weights.
    """
    cfg = make_config()
    buf = PrioritizedReplayBuffer(cfg)
    buf.batch_add([_exp() for _ in range(8)])

    # Same index repeated in one update batch (as PER sampling can produce).
    indices = np.array([3, 3, 5, 3])
    tds = np.array([2.0, 2.0, 1.0, 2.0])
    buf.update_priorities(indices, tds)

    leaf_start = buf.tree.capacity - 1
    leaf_sum = buf.tree.tree[leaf_start:leaf_start + buf.tree.capacity].sum()
    assert buf.tree.total() == pytest.approx(leaf_sum, rel=1e-12)

    expected = (2.0 + cfg['per']['epsilon']) ** cfg['per']['alpha']
    assert _leaf(buf, 3) == pytest.approx(expected, rel=1e-9)


def test_next_action_mask_stored_and_fallback():
    """Explicit next masks are stored; missing ones derive from next position."""
    cfg = make_config()
    buf = PrioritizedReplayBuffer(cfg)

    explicit = [1.0, 0.0, 1.0, 0.0]
    in_pos_next = {'position': [1.0, 0.0] + [0.0] * 8}   # is_long=1
    flat_next = {'position': [0.0] * 10}

    buf.batch_add([
        ({}, 2, 0.0, flat_next, False, None, explicit),   # explicit mask wins
        ({}, 2, 0.0, in_pos_next, False, None, None),     # fallback: in position
        ({}, 2, 0.0, flat_next, False, None, None),       # fallback: flat
        ({}, 2, 0.0, flat_next, False, None, [0, 0, 0, 0]),  # all-zero rejected
    ])

    nm = buf.next_action_masks
    assert nm[0].tolist() == explicit
    assert nm[1].tolist() == [0.0, 0.0, 1.0, 1.0]   # HOLD/CLOSE in position
    assert nm[2].tolist() == [1.0, 1.0, 1.0, 0.0]   # LONG/SHORT/HOLD when flat
    assert nm[3].tolist() == [1.0, 1.0, 1.0, 0.0]   # zero mask → fallback
