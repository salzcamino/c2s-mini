import pytest
import scanpy as sc
import numpy as np
from c2s_mini.data import C2SData
from c2s_mini.model import C2SModel
from c2s_mini.tasks import predict_cell_types, generate_cells, embed_cells


@pytest.fixture
def small_csdata():
    """Small C2SData object for testing."""
    adata = sc.datasets.pbmc3k()
    adata = adata[:5, :50]
    return C2SData.from_anndata(adata)


@pytest.mark.slow
def test_predict_cell_types(small_csdata):
    """Test cell type prediction task."""
    model = C2SModel(device='cpu')

    predictions = predict_cell_types(
        small_csdata,
        model,
        n_genes=50,
        batch_size=2,
        max_tokens=20
    )

    assert len(predictions) == len(small_csdata)
    assert all(isinstance(p, str) for p in predictions)


@pytest.mark.slow
def test_generate_cells():
    """Test cell generation task."""
    model = C2SModel(device='cpu')
    cell_types = ['B cell', 'T cell']

    generated = generate_cells(
        cell_types,
        model,
        n_genes=50,
        batch_size=2,
        max_tokens=100
    )

    assert len(generated) == len(cell_types)
    assert all(isinstance(g, str) for g in generated)


@pytest.mark.slow
def test_embed_cells(small_csdata):
    """Test cell embedding task."""
    model = C2SModel(device='cpu')

    embeddings = embed_cells(
        small_csdata,
        model,
        n_genes=50,
        batch_size=2
    )

    assert embeddings.shape[0] == len(small_csdata)
    assert embeddings.shape[1] > 0  # Has embedding dimension
    assert isinstance(embeddings, np.ndarray)
