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
