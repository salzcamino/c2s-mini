import pytest
import numpy as np
import scanpy as sc
from c2s_mini.utils import (
    generate_vocabulary,
    generate_sentences,
    reconstruct_expression_from_cell_sentence,
    post_process_generated_cell_sentences
)


@pytest.fixture
def small_adata():
    """Small test dataset."""
    adata = sc.datasets.pbmc3k()
    return adata[:10, :50]


def test_generate_vocabulary(small_adata):
    vocab = generate_vocabulary(small_adata)
    assert len(vocab) == small_adata.n_vars
    assert all(isinstance(k, str) for k in vocab.keys())
    assert all(k.isupper() for k in vocab.keys())


def test_generate_sentences(small_adata):
    vocab = generate_vocabulary(small_adata)
    sentences = generate_sentences(small_adata, vocab)

    assert len(sentences) == small_adata.n_obs
    assert all(isinstance(s, str) for s in sentences)

    # Check ranking (first gene should have highest expression)
    first_cell_genes = sentences[0].split()
    assert len(first_cell_genes) > 0


def test_post_process_generated_sentences():
    vocab_list = ["GENE1", "GENE2", "GENE3"]

    # Test with valid genes
    sentence = "GENE1 GENE2 GENE3"
    processed, n_replaced = post_process_generated_cell_sentences(sentence, vocab_list)
    assert n_replaced == 0
    assert len(processed) == 3

    # Test with invalid genes
    sentence = "GENE1 INVALID GENE2"
    processed, n_replaced = post_process_generated_cell_sentences(sentence, vocab_list)
    assert n_replaced == 1

    # Test with duplicates
    sentence = "GENE1 GENE2 GENE1"
    processed, n_replaced = post_process_generated_cell_sentences(sentence, vocab_list)
    assert processed.count("GENE1") == 1


def test_reconstruct_expression():
    vocab_list = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
    sentence = "GENE1 GENE3 GENE5"

    expr = reconstruct_expression_from_cell_sentence(
        sentence,
        delimiter=' ',
        vocab_list=vocab_list,
        slope=1.0,
        intercept=0.0
    )

    assert len(expr) == 5
    assert expr[0] > 0  # GENE1 expressed
    assert expr[1] == 0  # GENE2 not expressed
    assert expr[2] > 0  # GENE3 expressed


# Error handling and edge case tests


def test_post_process_empty_sentence():
    """Test post-processing with empty sentence."""
    vocab_list = ["GENE1", "GENE2"]
    sentence = ""
    processed, n_replaced = post_process_generated_cell_sentences(sentence, vocab_list)
    assert len(processed) == 1  # Empty string splits to ['']
    assert n_replaced == 1  # Empty string is invalid


def test_post_process_all_invalid():
    """Test post-processing when all genes are invalid."""
    vocab_list = ["GENE1", "GENE2"]
    sentence = "INVALID1 INVALID2 INVALID3"
    processed, n_replaced = post_process_generated_cell_sentences(sentence, vocab_list)
    assert n_replaced == 3
    assert all(gene == "NOT_A_GENE" for gene in processed)


def test_post_process_case_insensitive():
    """Test that post-processing handles lowercase genes."""
    vocab_list = ["GENE1", "GENE2", "GENE3"]
    sentence = "gene1 GENE2 Gene3"  # Mixed case
    processed, n_replaced = post_process_generated_cell_sentences(sentence, vocab_list)
    assert n_replaced == 0  # Should convert to uppercase and match
    assert "GENE1" in processed
    assert "GENE2" in processed
    assert "GENE3" in processed


def test_reconstruct_empty_sentence():
    """Test reconstruction with empty sentence."""
    vocab_list = ["GENE1", "GENE2", "GENE3"]
    sentence = ""
    expr = reconstruct_expression_from_cell_sentence(
        sentence,
        delimiter=' ',
        vocab_list=vocab_list,
        slope=1.0,
        intercept=0.0
    )
    assert len(expr) == 3
    assert all(e == 0 for e in expr)  # All zeros for empty sentence


def test_reconstruct_genes_not_in_vocab():
    """Test reconstruction with genes not in vocabulary."""
    vocab_list = ["GENE1", "GENE2", "GENE3"]
    sentence = "GENE1 INVALID GENE2"  # INVALID not in vocab
    expr = reconstruct_expression_from_cell_sentence(
        sentence,
        delimiter=' ',
        vocab_list=vocab_list,
        slope=1.0,
        intercept=0.0
    )
    assert len(expr) == 3
    assert expr[0] > 0  # GENE1 expressed
    assert expr[1] > 0  # GENE2 expressed
    # INVALID is ignored


def test_generate_vocabulary_single_gene(small_adata):
    """Test vocabulary generation with single gene."""
    single_gene_adata = small_adata[:, :1]
    vocab = generate_vocabulary(single_gene_adata)
    assert len(vocab) == 1


def test_generate_sentences_single_cell():
    """Test sentence generation with single cell."""
    adata = sc.datasets.pbmc3k()
    adata = adata[:1, :50]  # Just 1 cell
    vocab = generate_vocabulary(adata)
    sentences = generate_sentences(adata, vocab)
    assert len(sentences) == 1
    assert isinstance(sentences[0], str)


def test_post_process_multiple_duplicates():
    """Test handling of genes that appear many times."""
    vocab_list = ["GENE1", "GENE2", "GENE3"]
    sentence = "GENE1 GENE1 GENE1 GENE2 GENE1"
    processed, n_replaced = post_process_generated_cell_sentences(sentence, vocab_list)
    assert processed.count("GENE1") == 1  # Should only appear once
    assert "GENE2" in processed
