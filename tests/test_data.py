import pytest
import scanpy as sc
from c2s_mini.data import C2SData


@pytest.fixture
def small_adata():
    adata = sc.datasets.pbmc3k()
    adata = adata[:10, :50]
    adata.obs['cell_type'] = 'T cell'
    return adata


def test_csdata_from_anndata(small_adata):
    csdata = C2SData.from_anndata(small_adata)

    assert len(csdata) == small_adata.n_obs
    assert len(csdata.get_vocab()) == small_adata.n_vars
    assert len(csdata.get_sentences()) == small_adata.n_obs


def test_csdata_with_metadata(small_adata):
    csdata = C2SData.from_anndata(
        small_adata,
        include_obs_columns=['cell_type']
    )

    assert csdata.metadata is not None
    assert 'cell_type' in csdata.metadata.columns
    assert csdata.metadata['cell_type'].iloc[0] == 'T cell'


def test_csdata_to_dict(small_adata):
    csdata = C2SData.from_anndata(small_adata)
    data_dict = csdata.to_dict()

    assert 'sentences' in data_dict
    assert 'vocab' in data_dict
    assert len(data_dict['sentences']) == 10


# Error handling and edge case tests


def test_csdata_missing_metadata_column(small_adata):
    """Test error when requesting non-existent metadata column."""
    with pytest.raises(ValueError, match="not found in adata.obs"):
        C2SData.from_anndata(
            small_adata,
            include_obs_columns=['nonexistent_column']
        )


def test_csdata_multiple_missing_columns(small_adata):
    """Test error message lists all missing columns."""
    with pytest.raises(ValueError, match="not found in adata.obs"):
        C2SData.from_anndata(
            small_adata,
            include_obs_columns=['missing1', 'missing2', 'cell_type']
        )


def test_csdata_metadata_mismatch():
    """Test error when metadata length doesn't match sentences."""
    import pandas as pd
    from collections import OrderedDict

    vocab = OrderedDict([('GENE1', 10), ('GENE2', 5)])
    sentences = ['GENE1 GENE2', 'GENE2 GENE1']
    metadata = pd.DataFrame({'cell_type': ['T cell']})  # Only 1 row, but 2 sentences

    with pytest.raises(ValueError, match="Metadata has.*rows but there are.*sentences"):
        C2SData(vocab=vocab, sentences=sentences, metadata=metadata)


def test_csdata_empty_metadata(small_adata):
    """Test C2SData creation without metadata."""
    csdata = C2SData.from_anndata(small_adata, include_obs_columns=None)
    assert csdata.metadata is None


def test_csdata_string_representation(small_adata):
    """Test __str__ and __repr__ methods."""
    csdata = C2SData.from_anndata(small_adata)
    str_repr = str(csdata)
    assert "C2SData" in str_repr
    assert "n_cells=10" in str_repr
    assert "n_genes=50" in str_repr

    # Test repr
    repr_str = repr(csdata)
    assert repr_str == str_repr


def test_csdata_with_metadata_string(small_adata):
    """Test string representation includes metadata info."""
    csdata = C2SData.from_anndata(small_adata, include_obs_columns=['cell_type'])
    str_repr = str(csdata)
    assert "metadata_cols=1" in str_repr


def test_csdata_single_cell():
    """Test C2SData with single cell."""
    adata = sc.datasets.pbmc3k()
    adata = adata[:1, :50]
    csdata = C2SData.from_anndata(adata)
    assert len(csdata) == 1
    assert len(csdata.get_sentences()) == 1
