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
