"""Folding a tiled-input attribution back onto the record: no dataset or model needed."""
import numpy as np

from src.models import fit_length
from src.xai import fold, tile_index


def test_tile_index_matches_fit_length():
    rec = np.full((2, 20), np.nan)
    rec[0, 5:12] = np.arange(7.0)
    idx = tile_index(rec, 30)
    tiled = fit_length(rec, 30, "tile")
    assert np.array_equal(rec[0, idx[0]], tiled[0])
    assert len(idx[1]) == 0


def test_fold_keeps_the_total_and_marks_missing_samples():
    rec = np.full((2, 20), np.nan)
    rec[0, 5:12] = 1.0
    idx = tile_index(rec, 30)
    a = np.random.default_rng(0).random((2, 30))
    r = fold(a, idx, 20)
    assert np.isclose(np.nansum(r[0]), a[0].sum())
    assert np.isnan(r[0, :5]).all() and np.isnan(r[0, 12:]).all()
    assert np.isnan(r[1]).all()


def test_occlude_finds_the_lead_and_stretch_the_score_reads():
    from src.explain import occlude
    rec = np.full((3, 40), np.nan)
    rec[:, :20] = 1.0
    idx = tile_index(rec, 40)
    x = np.ones((3, 40), np.float32)
    x[0, 10:20] = x[0, 30:40] = 3.0                     # the score below reads lead 0, record samples 10 to 19 (both repeats)
    score = lambda xs: np.stack([xs[:, 0, 10:20].mean(axis=1) / 3], axis=1)
    blocks, lead_drop, windows = occlude(score, x, np.zeros_like(x), idx, 0, 10)
    assert np.argmax(lead_drop) == 0 and np.allclose(lead_drop[1:], 0)
    assert max(windows, key=lambda w: w[3])[:3] == (0, 10, 20)
    assert np.allclose(np.nansum(blocks[0, 10:20]), 1.0)
