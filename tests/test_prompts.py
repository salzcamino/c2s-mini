import pytest
from c2s_mini.prompts import (
    format_cell_type_prediction,
    format_cell_generation,
    truncate_sentence
)


def test_truncate_sentence():
    sentence = "GENE1 GENE2 GENE3 GENE4 GENE5"
    truncated = truncate_sentence(sentence, n_genes=3)
    assert truncated == "GENE1 GENE2 GENE3"


def test_format_cell_type_prediction():
    sentence = "CD3D CD3E CD8A"
    prompt = format_cell_type_prediction(sentence, n_genes=3)

    assert "3 genes" in prompt
    assert "CD3D CD3E CD8A" in prompt
    assert "Cell type:" in prompt


def test_format_cell_generation():
    prompt = format_cell_generation("B cell", n_genes=200)

    assert "B cell" in prompt
    assert "200 genes" in prompt
    assert "Genes:" in prompt


def test_organism_parameter():
    sentence = "CD3D CD3E"

    prompt_human = format_cell_type_prediction(sentence, organism='Homo sapiens')
    prompt_mouse = format_cell_type_prediction(sentence, organism='Mus musculus')

    assert "Homo sapiens" in prompt_human
    assert "Mus musculus" in prompt_mouse


# Error handling and edge case tests


def test_truncate_sentence_longer_than_genes():
    """Test truncation when n_genes is larger than available genes."""
    sentence = "GENE1 GENE2 GENE3"
    truncated = truncate_sentence(sentence, n_genes=10)
    assert truncated == "GENE1 GENE2 GENE3"  # Should return all genes


def test_truncate_empty_sentence():
    """Test truncation with empty sentence."""
    sentence = ""
    truncated = truncate_sentence(sentence, n_genes=5)
    assert truncated == ""


def test_truncate_sentence_zero_genes():
    """Test truncation with n_genes=0."""
    sentence = "GENE1 GENE2 GENE3"
    truncated = truncate_sentence(sentence, n_genes=0)
    assert truncated == ""


def test_truncate_single_gene():
    """Test truncation to single gene."""
    sentence = "GENE1 GENE2 GENE3"
    truncated = truncate_sentence(sentence, n_genes=1)
    assert truncated == "GENE1"


def test_format_cell_type_prediction_empty_sentence():
    """Test formatting with empty sentence."""
    sentence = ""
    prompt = format_cell_type_prediction(sentence, n_genes=10)
    assert "Genes:" in prompt
    assert "Cell type:" in prompt


def test_format_cell_generation_empty_cell_type():
    """Test formatting with empty cell type."""
    prompt = format_cell_generation("", n_genes=100)
    assert "single cell" in prompt
    assert "100 genes" in prompt


def test_format_cell_generation_complex_cell_type():
    """Test formatting with complex cell type name."""
    cell_type = "CD4+ T cell"
    prompt = format_cell_generation(cell_type, n_genes=50)
    assert "CD4+ T cell" in prompt
    assert "50 genes" in prompt


def test_truncate_custom_delimiter():
    """Test truncation with custom delimiter."""
    sentence = "GENE1|GENE2|GENE3|GENE4"
    truncated = truncate_sentence(sentence, n_genes=2, delimiter='|')
    assert truncated == "GENE1|GENE2"


def test_format_prediction_custom_organisms():
    """Test that custom organism names are included in prompt."""
    sentence = "GENE1 GENE2"
    prompt = format_cell_type_prediction(sentence, n_genes=2, organism="Custom organism")
    assert "Custom organism" in prompt
