import logging
from typing import TypedDict

logger = logging.getLogger(__name__)

_MS_PER_MONTH = 30 * 24 * 3600 * 1_000  # milliseconds


class Fold(TypedDict):
    train: list
    val: list


def build_walk_forward_folds(
    windows: list,
    train_months: float,
    val_months: float,
    test_months: float,
) -> tuple[list[Fold], list]:
    """Split windows into walk-forward folds + a held-out test block.

    Splitting is done right-to-left from the end of the data:
      1. Reserve the last `test_months` as test (never trained on).
      2. From the remaining IS data, carve alternating val/train blocks:
           val(val_months) + train(train_months)  — repeated from the end.
      3. Whatever is left at the start becomes the first train block.
      4. Edge case: if the first train block's time span < val_months,
         absorb it into the adjacent val block (train becomes empty).

    Returns
    -------
    folds      : list of Fold dicts sorted oldest-first
    test_windows: list of windows in the held-out test period
    """
    if not windows:
        return [], []

    # Use max future_close_time as the window's true "last timestamp" to prevent
    # label leakage: a window whose target extends into the test period must not
    # be assigned to train/val even if its entry candle is before the boundary.
    def _label_end(w: dict) -> int:
        fct = w.get('future_close_times')
        if fct is not None and len(fct):
            return int(max(fct))
        return int(w['current_close_time'])

    wins = sorted(windows, key=lambda w: w['current_close_time'])

    val_ms   = int(val_months   * _MS_PER_MONTH)
    train_ms = int(train_months * _MS_PER_MONTH)
    test_ms  = int(test_months  * _MS_PER_MONTH)

    t_last = _label_end(wins[-1])

    # ── Test block ───────────────────────────────────────────────────────────────
    test_boundary = t_last - test_ms
    test_w = [w for w in wins if _label_end(w) >  test_boundary]
    is_w   = [w for w in wins if _label_end(w) <= test_boundary]

    if not is_w:
        logger.warning('[WalkForward] No IS data after reserving test block.')
        return [], test_w

    # ── Build folds right-to-left ────────────────────────────────────────────────
    folds: list[Fold] = []
    cursor = test_boundary  # exclusive right boundary, moves left each iteration

    while True:
        val_end     = cursor
        val_start   = cursor - val_ms
        train_end   = val_start
        train_start = val_start - train_ms

        val_block   = [w for w in is_w if val_start   < _label_end(w) <= val_end]
        train_block = [w for w in is_w if train_start < _label_end(w) <= train_end]

        if not val_block:
            break

        folds.insert(0, Fold(train=train_block, val=val_block))
        cursor = train_start

        if not any(_label_end(w) <= cursor for w in is_w):
            break

    # ── Prepend any remainder (IS windows before the first fold) ─────────────────
    if folds:
        first_train_t_start = (
            _label_end(folds[0]['train'][0])
            if folds[0]['train']
            else _label_end(folds[0]['val'][0])
        )
        remainder = [w for w in is_w if _label_end(w) < first_train_t_start]
        if remainder:
            folds[0]['train'] = remainder + folds[0]['train']
    else:
        folds = [Fold(train=[], val=is_w)]

    # ── Edge case: first train too short → absorb into val ───────────────────────
    if folds[0]['train']:
        ft = folds[0]['train']
        span = _label_end(ft[-1]) - _label_end(ft[0])
        if span < val_ms:
            folds[0]['val']   = ft + folds[0]['val']
            folds[0]['train'] = []

    # ── Logging ──────────────────────────────────────────────────────────────────
    logger.info(f'[WalkForward] {len(folds)} fold(s), {len(test_w)} test windows')
    for i, f in enumerate(folds, 1):
        t_cnt = len(f['train'])
        v_cnt = len(f['val'])
        t_span = _span_months(f['train'])
        v_span = _span_months(f['val'])
        logger.info(
            f'  Fold {i}: train={t_cnt} ({t_span:.1f}M)  val={v_cnt} ({v_span:.1f}M)'
        )
    logger.info(f'  Test : {len(test_w)} windows ({_span_months(test_w):.1f}M)')

    return folds, test_w


def _span_months(windows: list) -> float:
    if not windows:
        return 0.0
    t0 = windows[0]['current_close_time']
    t1 = windows[-1]['current_close_time']
    return (t1 - t0) / _MS_PER_MONTH
