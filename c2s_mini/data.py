"""
Data wrapper class for c2s-mini.

This module provides the C2SData class, which wraps cell sentence data
and provides a clean API for working with cell sentences and metadata.

Reference:
    Original implementation: https://github.com/vandijklab/cell2sentence
"""

import sys
from collections import OrderedDict
from typing import Optional

import pandas as pd

from c2s_mini.utils import generate_vocabulary, generate_sentences


class C2SData:
    """
    Lightweight wrapper for cell sentence data.

    This class stores cell sentences, vocabulary, and optional metadata
    in-memory. Unlike the original Cell2Sentence implementation, this
    version does not use Arrow datasets or disk persistence.

    Attributes:
        vocab (OrderedDict): Ordered dictionary mapping gene names to expression counts.
        sentences (list): List of cell sentence strings.
        metadata (pd.DataFrame): Optional DataFrame with cell metadata.

    Example:
        >>> import scanpy as sc
        >>> from c2s_mini import C2SData
        >>>
        >>> # Load AnnData
        >>> adata = sc.datasets.pbmc3k()
        >>>
        >>> # Create C2SData
        >>> csdata = C2SData.from_anndata(adata)
        >>> print(csdata)
        >>>
        >>> # Access sentences
        >>> sentences = csdata.get_sentences()
        >>> print(sentences[0][:100])  # First 100 chars
    """

    def __init__(
        self,
        vocab: OrderedDict,
        sentences: list[str],
        metadata: Optional[pd.DataFrame] = None
    ):
        """
        Initialize C2SData object.

        Args:
            vocab: OrderedDict of gene names (uppercase) → expression counts.
            sentences: List of cell sentence strings.
            metadata: Optional DataFrame with cell metadata (one row per cell).
                Should have the same number of rows as len(sentences).

        Raises:
            ValueError: If metadata has different number of rows than sentences.
        """
        self.vocab = vocab
        self.sentences = sentences
        self.metadata = metadata

        # Validate metadata if provided
        if metadata is not None:
            if len(metadata) != len(sentences):
                raise ValueError(
                    f"Metadata has {len(metadata)} rows but there are "
                    f"{len(sentences)} sentences. They must match."
                )

    @classmethod
    def from_anndata(
        cls,
        adata,
        delimiter: str = ' ',
        random_state: int = 42,
        include_obs_columns: Optional[list[str]] = None
    ):
        """
        Create C2SData from AnnData object.

        This class method constructs a C2SData object from an AnnData object
        by generating vocabulary and cell sentences.

        Args:
            adata: AnnData object (obs=cells, vars=genes).
            delimiter: Gene separator in sentences (default: ' ').
            random_state: Random seed for tie-breaking in gene ranking (default: 42).
            include_obs_columns: Optional list of .obs column names to include
                in metadata. If None, no metadata is included.

        Returns:
            C2SData: New C2SData object containing cell sentences and metadata.

        Example:
            >>> import scanpy as sc
            >>> from c2s_mini import C2SData
            >>>
            >>> # Load data with annotations
            >>> adata = sc.datasets.pbmc3k_processed()
            >>>
            >>> # Create C2SData with cell type labels
            >>> csdata = C2SData.from_anndata(
            ...     adata,
            ...     include_obs_columns=['louvain']
            ... )
            >>>
            >>> # Access metadata
            >>> print(csdata.metadata.head())
        """
        # Warn if var_names contains Ensembl IDs instead of gene names
        if len(adata.var_names) > 0:
            first_gene_name = str(adata.var_names[0])
            if "ENS" in first_gene_name:
                print(
                    "WARN: adata.var_names seems to contain Ensembl IDs rather than "
                    "gene/feature names. It is highly recommended to use gene names "
                    "in cell sentences.",
                    file=sys.stderr
                )

        # Generate vocabulary and sentences
        vocab = generate_vocabulary(adata)
        sentences = generate_sentences(adata, vocab, delimiter=delimiter, random_state=random_state)

        # Optionally extract metadata
        metadata = None
        if include_obs_columns is not None:
            # Validate that all requested columns exist
            missing_cols = [col for col in include_obs_columns if col not in adata.obs.columns]
            if missing_cols:
                raise ValueError(
                    f"Requested columns {missing_cols} not found in adata.obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

            # Extract the requested columns
            metadata = adata.obs[include_obs_columns].copy()
            # Reset index to simple integer index for consistency
            metadata = metadata.reset_index(drop=True)

        return cls(vocab=vocab, sentences=sentences, metadata=metadata)

    def get_sentences(self) -> list[str]:
        """
        Return list of cell sentences.

        Returns:
            list: List of cell sentence strings, one per cell.

        Example:
            >>> sentences = csdata.get_sentences()
            >>> print(f"Number of cells: {len(sentences)}")
            >>> print(f"First sentence: {sentences[0][:80]}...")
        """
        return self.sentences

    def get_vocab(self) -> OrderedDict:
        """
        Return vocabulary dictionary.

        Returns:
            OrderedDict: Ordered dictionary mapping gene names (uppercase)
                to the number of cells in which they are expressed.

        Example:
            >>> vocab = csdata.get_vocab()
            >>> print(f"Number of genes: {len(vocab)}")
            >>> print(f"First 5 genes: {list(vocab.keys())[:5]}")
        """
        return self.vocab

    def to_dict(self) -> dict:
        """
        Convert to dictionary format.

        Returns:
            dict: Dictionary with keys:
                - 'sentences': list of cell sentence strings
                - 'vocab': OrderedDict of gene names → counts
                - 'metadata': DataFrame of metadata (if present, else None)

        Example:
            >>> data_dict = csdata.to_dict()
            >>> print(data_dict.keys())
            >>> # Save to JSON (after converting vocab to regular dict)
            >>> import json
            >>> json_dict = {
            ...     'sentences': data_dict['sentences'],
            ...     'vocab': dict(data_dict['vocab']),
            ... }
        """
        return {
            'sentences': self.sentences,
            'vocab': self.vocab,
            'metadata': self.metadata
        }

    def __len__(self) -> int:
        """
        Return number of cells.

        Returns:
            int: Number of cells (i.e., number of sentences).

        Example:
            >>> print(f"Dataset contains {len(csdata)} cells")
        """
        return len(self.sentences)

    def __str__(self) -> str:
        """
        String representation of C2SData object.

        Returns:
            str: Human-readable summary of the object.

        Example:
            >>> print(csdata)
            C2SData(n_cells=2700, n_genes=1838)
        """
        n_cells = len(self.sentences)
        n_genes = len(self.vocab)
        has_metadata = self.metadata is not None

        if has_metadata:
            n_metadata_cols = len(self.metadata.columns)
            return (
                f"C2SData(n_cells={n_cells}, n_genes={n_genes}, "
                f"metadata_cols={n_metadata_cols})"
            )
        else:
            return f"C2SData(n_cells={n_cells}, n_genes={n_genes})"

    def __repr__(self) -> str:
        """
        Official string representation of C2SData object.

        Returns:
            str: String that could be used to recreate the object.
        """
        return self.__str__()
