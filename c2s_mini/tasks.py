"""
High-level task functions for Cell2Sentence operations.

This module provides user-friendly API functions for common Cell2Sentence tasks:
- Cell type prediction
- Cell generation conditioned on cell type
- Cell embedding extraction

Reference:
    Original implementation: https://github.com/vandijklab/cell2sentence
"""

import numpy as np
from tqdm import tqdm
from typing import Optional, List

from c2s_mini.data import C2SData
from c2s_mini.model import C2SModel
from c2s_mini.prompts import (
    format_cell_type_prediction,
    format_cell_generation,
    format_batch_cell_type_prediction,
    format_batch_cell_generation
)


def predict_cell_types(
    csdata: C2SData,
    model: C2SModel,
    n_genes: int = 200,
    organism: str = 'Homo sapiens',
    batch_size: int = 8,
    **generation_kwargs
) -> List[str]:
    """
    Predict cell types for all cells in dataset.

    This function takes a C2SData object containing cell sentences and uses
    the C2SModel to predict the cell type for each cell based on its top
    expressed genes.

    Args:
        csdata: C2SData object with cell sentences
        model: C2SModel for inference
        n_genes: Number of genes to use per cell (default: 200)
        organism: 'Homo sapiens' or 'Mus musculus' (default: 'Homo sapiens')
        batch_size: Batch size for inference (default: 8)
        **generation_kwargs: Additional args for model.generate_batch()
            (e.g., max_tokens, temperature, top_p)

    Returns:
        List of predicted cell type strings (one per cell)

    Example:
        >>> import scanpy as sc
        >>> from c2s_mini import C2SData, C2SModel, predict_cell_types
        >>>
        >>> # Load data
        >>> adata = sc.datasets.pbmc3k()[:100]
        >>> csdata = C2SData.from_anndata(adata)
        >>>
        >>> # Load model and predict
        >>> model = C2SModel(device='cpu')
        >>> predictions = predict_cell_types(
        ...     csdata, model, n_genes=100, batch_size=4
        ... )
        >>> print(predictions[:5])
    """
    sentences = csdata.get_sentences()
    predictions = []

    print(f"Predicting cell types for {len(sentences)} cells...")

    # Process in batches with progress bar
    for i in tqdm(range(0, len(sentences), batch_size), desc="Predicting"):
        batch = sentences[i:i + batch_size]
        prompts = format_batch_cell_type_prediction(batch, n_genes, organism)
        batch_predictions = model.generate_batch(prompts, **generation_kwargs)
        predictions.extend(batch_predictions)

    return predictions


def generate_cells(
    cell_types: List[str],
    model: C2SModel,
    n_genes: int = 200,
    organism: str = 'Homo sapiens',
    batch_size: int = 8,
    **generation_kwargs
) -> List[str]:
    """
    Generate cell sentences conditioned on cell types.

    This function uses the C2SModel to generate synthetic cell sentences for
    specified cell types. The model generates a ranked list of genes that
    would be characteristic of each cell type.

    Args:
        cell_types: List of cell type labels to generate
        model: C2SModel for inference
        n_genes: Number of genes to generate per cell (default: 200)
        organism: 'Homo sapiens' or 'Mus musculus' (default: 'Homo sapiens')
        batch_size: Batch size for inference (default: 8)
        **generation_kwargs: Additional args for model.generate_batch()
            (e.g., max_tokens, temperature, top_p)

    Returns:
        List of generated cell sentence strings (one per cell type)

    Example:
        >>> from c2s_mini import C2SModel, generate_cells
        >>>
        >>> # Define cell types to generate
        >>> cell_types = ["B cell", "T cell", "NK cell"]
        >>>
        >>> # Load model and generate
        >>> model = C2SModel(device='cpu')
        >>> generated = generate_cells(
        ...     cell_types, model, n_genes=100, batch_size=3
        ... )
        >>> print(f"Generated {len(generated)} cells")
        >>> print(f"B cell genes: {generated[0][:80]}...")
    """
    generated = []

    print(f"Generating {len(cell_types)} cells...")

    # Process in batches with progress bar
    for i in tqdm(range(0, len(cell_types), batch_size), desc="Generating"):
        batch = cell_types[i:i + batch_size]
        prompts = format_batch_cell_generation(batch, n_genes, organism)
        batch_generated = model.generate_batch(prompts, **generation_kwargs)
        generated.extend(batch_generated)

    return generated


def embed_cells(
    csdata: C2SData,
    model: C2SModel,
    n_genes: int = 200,
    organism: str = 'Homo sapiens',
    batch_size: int = 8
) -> np.ndarray:
    """
    Generate embeddings for all cells in dataset.

    This function extracts embedding vectors from the C2SModel for each cell
    in the dataset. The embeddings capture a learned representation of each
    cell's expression profile and can be used for downstream tasks like
    clustering, visualization, or similarity search.

    Args:
        csdata: C2SData object with cell sentences
        model: C2SModel for inference
        n_genes: Number of genes to use per cell (default: 200)
        organism: 'Homo sapiens' or 'Mus musculus' (default: 'Homo sapiens')
        batch_size: Batch size for inference (default: 8)

    Returns:
        Array of embeddings with shape (n_cells, embedding_dim)

    Example:
        >>> import scanpy as sc
        >>> from c2s_mini import C2SData, C2SModel, embed_cells
        >>>
        >>> # Load data
        >>> adata = sc.datasets.pbmc3k()[:100]
        >>> csdata = C2SData.from_anndata(adata)
        >>>
        >>> # Load model and embed
        >>> model = C2SModel(device='cpu')
        >>> embeddings = embed_cells(csdata, model, n_genes=100, batch_size=8)
        >>> print(f"Embeddings shape: {embeddings.shape}")
        >>>
        >>> # Use embeddings for UMAP visualization
        >>> import umap
        >>> reducer = umap.UMAP()
        >>> umap_coords = reducer.fit_transform(embeddings)
    """
    sentences = csdata.get_sentences()
    all_embeddings = []

    print(f"Embedding {len(sentences)} cells...")

    # Process in batches with progress bar
    for i in tqdm(range(0, len(sentences), batch_size), desc="Embedding"):
        batch = sentences[i:i + batch_size]
        # Use prediction format for embedding prompts
        prompts = format_batch_cell_type_prediction(batch, n_genes, organism)
        batch_embeddings = model.embed_batch(prompts)
        all_embeddings.append(batch_embeddings)

    # Stack all batch embeddings into a single array
    return np.vstack(all_embeddings)
