"""
c2s-mini: Miniature Cell2Sentence implementation

A lightweight implementation of Cell2Sentence for single-cell analysis with LLMs.
Supports inference with the pythia-160m-c2s model for cell type prediction,
cell generation, and cell embedding tasks.

Example usage:
    >>> from c2s_mini import C2SData, C2SModel, predict_cell_types
    >>> import scanpy as sc
    >>>
    >>> # Load data
    >>> adata = sc.datasets.pbmc3k()
    >>> csdata = C2SData.from_anndata(adata)
    >>>
    >>> # Load model and predict
    >>> model = C2SModel()
    >>> predictions = predict_cell_types(csdata, model, n_genes=100)

Reference:
    Original Cell2Sentence: https://github.com/vandijklab/cell2sentence
    Paper: https://www.biorxiv.org/content/10.1101/2023.09.11.557287
"""

__version__ = "0.1.0"
__author__ = "c2s-mini contributors"
__license__ = "Apache-2.0"

# Core components
# Phase 1 (utils.py) - Complete
# Phase 2 (data.py) - Complete
# Phase 3 (model.py) - Complete
# Phase 4 (prompts.py) - Complete
# Phase 5 (tasks.py) - Complete

from .data import C2SData
from .model import C2SModel
from .tasks import predict_cell_types, generate_cells, embed_cells

__all__ = [
    "C2SData",
    "C2SModel",
    "predict_cell_types",
    "generate_cells",
    "embed_cells",
]
