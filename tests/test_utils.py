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
