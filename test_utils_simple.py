#!/usr/bin/env python3
"""
Simple unit tests for utils.py that don't require scanpy.
"""

import numpy as np
from scipy import sparse
from collections import OrderedDict

# Test imports
print("Testing imports...")
try:
    from c2s_mini.utils import (
        sort_transcript_counts,
        post_process_generated_cell_sentences,
        reconstruct_expression_from_cell_sentence
    )
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 1: sort_transcript_counts
print("\nTest 1: sort_transcript_counts")
try:
    test_expr = np.array([[5, 2, 8, 1], [3, 9, 3, 0]])
    ranks = sort_transcript_counts(test_expr)
    assert ranks.shape == test_expr.shape, "Shape mismatch"
    # Highest value (8 at index 2) should get rank 0
    assert ranks[0, 2] == 0, f"Expected rank 0 for highest value, got {ranks[0, 2]}"
    # Second highest (5 at index 0) should get rank 1
    assert ranks[0, 0] == 1, f"Expected rank 1 for second highest, got {ranks[0, 0]}"
    print(f"✓ sort_transcript_counts works correctly")
    print(f"  Input:  {test_expr[0]}")
    print(f"  Ranks:  {ranks[0].astype(int)}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 2: post_process_generated_cell_sentences with valid genes
print("\nTest 2: post_process_generated_cell_sentences (valid genes)")
try:
    vocab_list = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

    # Test with valid genes
    test_sentence = "GENE1 GENE2 GENE3"
    processed, n_replaced = post_process_generated_cell_sentences(
        test_sentence, vocab_list
    )
    assert n_replaced == 0, f"Should not replace valid genes, but replaced {n_replaced}"
    assert len(processed) == 3, f"Should have 3 genes, got {len(processed)}"
    print(f"✓ Valid genes test passed (0 replaced)")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 3: post_process_generated_cell_sentences with invalid genes
print("\nTest 3: post_process_generated_cell_sentences (invalid genes)")
try:
    vocab_list = ["GENE1", "GENE2", "GENE3"]
    test_sentence = "GENE1 INVALID_GENE GENE2"
    processed, n_replaced = post_process_generated_cell_sentences(
        test_sentence, vocab_list
    )
    assert n_replaced == 1, f"Should replace 1 invalid gene, replaced {n_replaced}"
    assert "NOT_A_GENE" in processed, "Should contain placeholder for invalid gene"
    print(f"✓ Invalid genes test passed (1 replaced)")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 4: post_process_generated_cell_sentences with duplicates
print("\nTest 4: post_process_generated_cell_sentences (duplicates)")
try:
    vocab_list = ["GENE1", "GENE2", "GENE3"]
    test_sentence = "GENE1 GENE2 GENE1"
    processed, n_replaced = post_process_generated_cell_sentences(
        test_sentence, vocab_list
    )
    assert processed.count("GENE1") == 1, "Should deduplicate GENE1"
    assert len(processed) == 2, f"Should have 2 unique genes, got {len(processed)}"
    print(f"✓ Deduplication test passed")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 5: reconstruct_expression_from_cell_sentence
print("\nTest 5: reconstruct_expression_from_cell_sentence")
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

    assert len(expr) == len(vocab_list), f"Expression vector length mismatch: {len(expr)} != {len(vocab_list)}"
    assert expr[0] > 0, "GENE1 should be expressed"
    assert expr[1] == 0, "GENE2 should not be expressed"
    assert expr[2] > 0, "GENE3 should be expressed"
    assert expr[0] > expr[2], "GENE1 (rank 0) should have higher expression than GENE3 (rank 1)"
    print(f"✓ Expression reconstruction works correctly")
    print(f"  Input sentence: {sentence}")
    print(f"  Expression: {expr}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✓ All simple unit tests passed!")
print("=" * 60)
