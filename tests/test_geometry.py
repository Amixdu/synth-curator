"""core/geometry.py::knn_cosine - hand-computed cosine distances."""
import numpy as np
import pytest
import torch

from synth_curator.core.geometry import knn_cosine


def test_knn_cosine_hand_computed():
    # query along +x; db has one identical direction, one orthogonal, one opposite.
    query = torch.tensor([[1.0, 0.0]])
    db = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    d, i = knn_cosine(query, db, k=3)
    # squared L2 on normalized vectors = 2 - 2*cos: identical->0, orthogonal->2, opposite->4
    assert d[0].tolist() == pytest.approx([0.0, 2.0, 4.0], abs=1e-6)
    assert i[0].tolist() == [0, 1, 2]      # ascending distance order


def test_knn_cosine_k_less_than_db_size_returns_nearest_k_only():
    query = torch.tensor([[1.0, 0.0]])
    db = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    d, i = knn_cosine(query, db, k=1)
    assert d.shape == (1, 1) and i[0, 0] == 0


def test_knn_cosine_is_scale_invariant():
    """Normalization means magnitude never matters, only direction."""
    query = torch.tensor([[10.0, 0.0]])
    db = torch.tensor([[1.0, 0.0], [0.0, 100.0]])
    d, i = knn_cosine(query, db, k=2)
    assert i[0].tolist() == [0, 1]
    assert d[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_knn_cosine_chunking_matches_unchunked():
    rng = np.random.default_rng(0)
    query = torch.tensor(rng.normal(size=(20, 8)), dtype=torch.float32)
    db = torch.tensor(rng.normal(size=(50, 8)), dtype=torch.float32)
    d_full, i_full = knn_cosine(query, db, k=5, chunk=2048)
    d_small, i_small = knn_cosine(query, db, k=5, chunk=3)     # forces multiple chunks
    assert np.allclose(d_full, d_small, atol=1e-5)
    assert (i_full == i_small).all()
