#!/usr/bin/env python3
"""
Quick test script for Phase 1 utils.py implementation.
This will be replaced by proper pytest tests in Phase 7.
"""

import numpy as np
import scanpy as sc
from c2s_mini.utils import (
    generate_vocabulary,
    generate_sentences,
    sort_transcript_counts,
    post_process_generated_cell_sentences,
    reconstruct_expression_from_cell_sentence
)

print("=" * 60)
print("Testing Phase 1: Core Data Transformation (utils.py)")
print("=" * 60)

# Test 1: Load test data
print("\n[1/6] Loading test data...")
try:
    adata = sc.datasets.pbmc3k()
    adata = adata[:50, :200]  # Small subset for quick testing
    print(f"✓ Loaded {adata.n_obs} cells × {adata.n_vars} genes")
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    exit(1)

# Test 2: Generate vocabulary
print("\n[2/6] Testing generate_vocabulary()...")
try:
    vocab = generate_vocabulary(adata)
    assert len(vocab) == adata.n_vars, "Vocabulary size mismatch"
    assert all(isinstance(k, str) for k in vocab.keys()), "Keys not strings"
    assert all(k.isupper() for k in vocab.keys()), "Keys not uppercase"
    print(f"✓ Generated vocabulary with {len(vocab)} genes")
    print(f"  First 5 genes: {list(vocab.keys())[:5]}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 3: Generate sentences
print("\n[3/6] Testing generate_sentences()...")
try:
    sentences = generate_sentences(adata, vocab)
    assert len(sentences) == adata.n_obs, "Number of sentences mismatch"
    assert all(isinstance(s, str) for s in sentences), "Sentences not strings"
    print(f"✓ Generated {len(sentences)} cell sentences")
    print(f"  First sentence (first 80 chars): {sentences[0][:80]}...")
    print(f"  Sentence length: {len(sentences[0].split())} genes")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 4: Sort transcript counts
print("\n[4/6] Testing sort_transcript_counts()...")
try:
    # Create small test matrix
    test_expr = np.array([[5, 2, 8, 1], [3, 9, 3, 0]])
    ranks = sort_transcript_counts(test_expr)
    assert ranks.shape == test_expr.shape, "Shape mismatch"
    # Highest value should get rank 0
    assert ranks[0, 2] == 0, "Highest value should have rank 0"
    print(f"✓ Correctly ranked transcript counts")
    print(f"  Test expression: {test_expr[0]}")
    print(f"  Ranks: {ranks[0]}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 5: Post-process generated sentences
print("\n[5/6] Testing post_process_generated_cell_sentences()...")
try:
    vocab_list = list(vocab.keys())

    # Test with valid genes
    test_sentence = " ".join(vocab_list[:10])
    processed, n_replaced = post_process_generated_cell_sentences(
        test_sentence, vocab_list
    )
    assert n_replaced == 0, "Should not replace valid genes"
    assert len(processed) == 10, "Should preserve gene count"

    # Test with invalid genes
    test_sentence_invalid = f"{vocab_list[0]} NOTGENE {vocab_list[1]}"
    processed, n_replaced = post_process_generated_cell_sentences(
        test_sentence_invalid, vocab_list
    )
    assert n_replaced == 1, "Should replace invalid gene"

    # Test with duplicates
    test_sentence_dup = f"{vocab_list[0]} {vocab_list[1]} {vocab_list[0]}"
    processed, n_replaced = post_process_generated_cell_sentences(
        test_sentence_dup, vocab_list
    )
    assert processed.count(vocab_list[0]) == 1, "Should deduplicate genes"

    print(f"✓ Post-processing works correctly")
    print(f"  Valid genes test: {n_replaced} genes replaced")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 6: Reconstruct expression
print("\n[6/6] Testing reconstruct_expression_from_cell_sentence()...")
try:
    vocab_list = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
    sentence = "GENE1 GENE3 GENE5"

    expr = reconstruct_expression_from_cell_sentence(
        sentence,
        delimiter=' ',
        vocab_list=vocab_list,
        slope=-1.0,
        intercept=5.0
    )

    assert len(expr) == len(vocab_list), "Expression vector length mismatch"
    assert expr[0] > 0, "GENE1 should be expressed"
    assert expr[1] == 0, "GENE2 should not be expressed"
    assert expr[2] > 0, "GENE3 should be expressed"
    assert expr[0] > expr[2], "GENE1 (rank 0) should have higher expression than GENE3 (rank 1)"

    print(f"✓ Expression reconstruction works correctly")
    print(f"  Input sentence: {sentence}")
    print(f"  Reconstructed expression: {expr}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✓ All Phase 1 tests passed!")
print("=" * 60)
