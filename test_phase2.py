#!/usr/bin/env python3
"""
Integration tests for Phase 2: C2SData implementation.
Requires scanpy and other dependencies to be installed.
"""

import numpy as np
import scanpy as sc
import pandas as pd
from c2s_mini.data import C2SData

print("=" * 60)
print("Testing Phase 2: Data Wrapper (data.py)")
print("=" * 60)

# Test 1: Load test data
print("\n[1/7] Loading test data...")
try:
    adata = sc.datasets.pbmc3k()
    adata = adata[:100, :200]  # Small subset for quick testing
    print(f"✓ Loaded {adata.n_obs} cells × {adata.n_vars} genes")
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    exit(1)

# Test 2: Basic from_anndata without metadata
print("\n[2/7] Testing C2SData.from_anndata() without metadata...")
try:
    csdata = C2SData.from_anndata(adata)

    assert len(csdata) == adata.n_obs, f"Cell count mismatch: {len(csdata)} != {adata.n_obs}"
    assert len(csdata.get_vocab()) == adata.n_vars, f"Gene count mismatch"
    assert len(csdata.get_sentences()) == adata.n_obs, "Sentence count mismatch"
    assert csdata.metadata is None, "Metadata should be None when not requested"

    print(f"✓ Created C2SData without metadata")
    print(f"  {csdata}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 3: from_anndata with metadata
print("\n[3/7] Testing C2SData.from_anndata() with metadata...")
try:
    # Add a mock cell type annotation
    adata.obs['mock_cell_type'] = ['Type_A' if i % 2 == 0 else 'Type_B' for i in range(adata.n_obs)]

    csdata = C2SData.from_anndata(adata, include_obs_columns=['mock_cell_type'])

    assert csdata.metadata is not None, "Metadata should be present"
    assert len(csdata.metadata) == adata.n_obs, "Metadata row count mismatch"
    assert 'mock_cell_type' in csdata.metadata.columns, "Should have mock_cell_type column"
    assert csdata.metadata['mock_cell_type'].iloc[0] == 'Type_A', "First cell should be Type_A"
    assert csdata.metadata['mock_cell_type'].iloc[1] == 'Type_B', "Second cell should be Type_B"

    print(f"✓ Created C2SData with metadata")
    print(f"  {csdata}")
    print(f"  Metadata columns: {list(csdata.metadata.columns)}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 4: from_anndata with multiple metadata columns
print("\n[4/7] Testing from_anndata() with multiple metadata columns...")
try:
    # Add another annotation
    adata.obs['batch'] = [f"batch_{i%3}" for i in range(adata.n_obs)]

    csdata = C2SData.from_anndata(
        adata,
        include_obs_columns=['mock_cell_type', 'batch']
    )

    assert len(csdata.metadata.columns) == 2, "Should have 2 metadata columns"
    assert 'mock_cell_type' in csdata.metadata.columns, "Should have mock_cell_type"
    assert 'batch' in csdata.metadata.columns, "Should have batch"

    print(f"✓ Created C2SData with multiple metadata columns")
    print(f"  Metadata columns: {list(csdata.metadata.columns)}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 5: from_anndata with invalid column name
print("\n[5/7] Testing from_anndata() with invalid column name...")
try:
    try:
        csdata = C2SData.from_anndata(
            adata,
            include_obs_columns=['nonexistent_column']
        )
        print("✗ Should have raised ValueError for invalid column")
        exit(1)
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:70]}...")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 6: Check sentence content
print("\n[6/7] Testing sentence content and structure...")
try:
    csdata = C2SData.from_anndata(adata)
    sentences = csdata.get_sentences()
    vocab = csdata.get_vocab()

    # Check that sentences contain valid gene names
    first_sentence_genes = sentences[0].split()
    assert all(gene in vocab for gene in first_sentence_genes), \
        "All genes in sentences should be in vocabulary"

    # Check that sentences are non-empty for non-empty cells
    assert all(len(s) > 0 for s in sentences if s), \
        "Sentences should not be empty (unless cell has no expression)"

    # Check that vocabulary is ordered
    vocab_list = list(vocab.keys())
    assert all(isinstance(g, str) for g in vocab_list), \
        "All vocab keys should be strings"
    assert all(g.isupper() for g in vocab_list), \
        "All gene names should be uppercase"

    print(f"✓ Sentence content is valid")
    print(f"  First sentence (first 100 chars): {sentences[0][:100]}...")
    print(f"  Number of genes in first sentence: {len(first_sentence_genes)}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 7: Test to_dict() method
print("\n[7/7] Testing to_dict() method...")
try:
    csdata = C2SData.from_anndata(adata, include_obs_columns=['mock_cell_type'])
    data_dict = csdata.to_dict()

    assert 'sentences' in data_dict, "Should have 'sentences' key"
    assert 'vocab' in data_dict, "Should have 'vocab' key"
    assert 'metadata' in data_dict, "Should have 'metadata' key"

    assert len(data_dict['sentences']) == len(csdata), "Sentence count should match"
    assert len(data_dict['vocab']) == len(csdata.get_vocab()), "Vocab size should match"
    assert data_dict['metadata'] is not None, "Metadata should be present"

    print(f"✓ to_dict() returns correct structure")
    print(f"  Keys: {list(data_dict.keys())}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✓ All Phase 2 integration tests passed!")
print("=" * 60)
