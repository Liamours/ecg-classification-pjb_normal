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
