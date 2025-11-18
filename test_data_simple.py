#!/usr/bin/env python3
"""
Simple unit tests for data.py that work with minimal dependencies.
Tests C2SData class functionality.
"""

from collections import OrderedDict
import sys

# Test imports
print("Testing imports...")
try:
    from c2s_mini.data import C2SData
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 1: Basic constructor
print("\nTest 1: C2SData constructor")
try:
    vocab = OrderedDict([("GENE1", 10), ("GENE2", 8), ("GENE3", 5)])
    sentences = ["GENE1 GENE2", "GENE2 GENE3", "GENE1 GENE3"]

    csdata = C2SData(vocab=vocab, sentences=sentences)

    assert len(csdata) == 3, "Length should be 3"
    assert len(csdata.get_vocab()) == 3, "Vocab size should be 3"
    assert len(csdata.get_sentences()) == 3, "Should have 3 sentences"

    print("✓ Basic constructor works")
    print(f"  {csdata}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Constructor with metadata
print("\nTest 2: C2SData constructor with metadata")
try:
    import pandas as pd

    vocab = OrderedDict([("GENE1", 10), ("GENE2", 8)])
    sentences = ["GENE1 GENE2", "GENE2 GENE1"]
    metadata = pd.DataFrame({'cell_type': ['T cell', 'B cell']})

    csdata = C2SData(vocab=vocab, sentences=sentences, metadata=metadata)

    assert csdata.metadata is not None, "Metadata should be set"
    assert len(csdata.metadata) == 2, "Metadata should have 2 rows"
    assert 'cell_type' in csdata.metadata.columns, "Should have cell_type column"

    print("✓ Constructor with metadata works")
    print(f"  {csdata}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Constructor validation (mismatched lengths)
print("\nTest 3: Constructor validation")
try:
    import pandas as pd

    vocab = OrderedDict([("GENE1", 10)])
    sentences = ["GENE1", "GENE1"]
    metadata = pd.DataFrame({'cell_type': ['T cell']})  # Only 1 row, but 2 sentences

    try:
        csdata = C2SData(vocab=vocab, sentences=sentences, metadata=metadata)
        print("✗ Should have raised ValueError for mismatched lengths")
        sys.exit(1)
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:60]}...")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: get_sentences()
print("\nTest 4: get_sentences()")
try:
    vocab = OrderedDict([("GENE1", 10), ("GENE2", 8)])
    sentences = ["GENE1 GENE2", "GENE2 GENE1"]

    csdata = C2SData(vocab=vocab, sentences=sentences)
    retrieved = csdata.get_sentences()

    assert retrieved == sentences, "Retrieved sentences should match input"
    assert retrieved is sentences, "Should return same object"

    print("✓ get_sentences() works correctly")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 5: get_vocab()
print("\nTest 5: get_vocab()")
try:
    vocab = OrderedDict([("GENE1", 10), ("GENE2", 8), ("GENE3", 5)])
    sentences = ["GENE1 GENE2 GENE3"]

    csdata = C2SData(vocab=vocab, sentences=sentences)
    retrieved = csdata.get_vocab()

    assert retrieved == vocab, "Retrieved vocab should match input"
    assert retrieved is vocab, "Should return same object"

    print("✓ get_vocab() works correctly")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 6: to_dict()
print("\nTest 6: to_dict()")
try:
    import pandas as pd

    vocab = OrderedDict([("GENE1", 10), ("GENE2", 8)])
    sentences = ["GENE1 GENE2", "GENE2 GENE1"]
    metadata = pd.DataFrame({'cell_type': ['T cell', 'B cell']})

    csdata = C2SData(vocab=vocab, sentences=sentences, metadata=metadata)
    data_dict = csdata.to_dict()

    assert 'sentences' in data_dict, "Should have 'sentences' key"
    assert 'vocab' in data_dict, "Should have 'vocab' key"
    assert 'metadata' in data_dict, "Should have 'metadata' key"

    assert data_dict['sentences'] == sentences, "Sentences should match"
    assert data_dict['vocab'] == vocab, "Vocab should match"
    assert data_dict['metadata'].equals(metadata), "Metadata should match"

    print("✓ to_dict() works correctly")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 7: __len__()
print("\nTest 7: __len__()")
try:
    vocab = OrderedDict([("GENE1", 10)])
    sentences = ["GENE1"] * 5

    csdata = C2SData(vocab=vocab, sentences=sentences)

    assert len(csdata) == 5, "Length should be 5"

    print("✓ __len__() works correctly")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 8: __str__() and __repr__()
print("\nTest 8: __str__() and __repr__()")
try:
    import pandas as pd

    # Without metadata
    vocab = OrderedDict([("GENE1", 10), ("GENE2", 8)])
    sentences = ["GENE1 GENE2"] * 3
    csdata = C2SData(vocab=vocab, sentences=sentences)

    str_repr = str(csdata)
    assert "n_cells=3" in str_repr, "Should show cell count"
    assert "n_genes=2" in str_repr, "Should show gene count"
    assert "metadata_cols" not in str_repr, "Should not show metadata cols without metadata"

    # With metadata
    metadata = pd.DataFrame({'cell_type': ['T cell'] * 3})
    csdata2 = C2SData(vocab=vocab, sentences=sentences, metadata=metadata)

    str_repr2 = str(csdata2)
    assert "metadata_cols=1" in str_repr2, "Should show metadata column count"

    print("✓ __str__() and __repr__() work correctly")
    print(f"  Without metadata: {str_repr}")
    print(f"  With metadata: {str_repr2}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All C2SData unit tests passed!")
print("=" * 60)
