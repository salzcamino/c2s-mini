"""
Prompt formatting functions for Cell2Sentence tasks.

This module provides functions to format cell sentences and cell types into
prompts suitable for the Cell2Sentence language model.
"""


def truncate_sentence(sentence: str, n_genes: int, delimiter: str = ' ') -> str:
    """
    Truncate cell sentence to first n_genes.

    Args:
        sentence: Full cell sentence
        n_genes: Number of genes to keep
        delimiter: Gene separator

    Returns:
        Truncated sentence
    """
    genes = sentence.split(delimiter)
    return delimiter.join(genes[:n_genes])


def format_cell_type_prediction(
    cell_sentence: str,
    n_genes: int = 200,
    organism: str = 'Homo sapiens'
) -> str:
    """
    Format prompt for cell type prediction.

    Args:
        cell_sentence: Space-separated gene names
        n_genes: Number of genes to use
        organism: 'Homo sapiens' or 'Mus musculus'

    Returns:
        Formatted prompt string
    """
    truncated = truncate_sentence(cell_sentence, n_genes)

    # Template based on original single_cell_cell_type_prediction_prompts.json
    prompt = (
        f"Given the following list of {n_genes} genes ranked by expression "
        f"in a {organism} single cell, predict the cell type.\n\n"
        f"Genes: {truncated}\n\n"
        f"Cell type:"
    )
    return prompt


def format_cell_generation(
    cell_type: str,
    n_genes: int = 200,
    organism: str = 'Homo sapiens'
) -> str:
    """
    Format prompt for conditional cell generation.

    Args:
        cell_type: Target cell type to generate
        n_genes: Number of genes to generate
        organism: 'Homo sapiens' or 'Mus musculus'

    Returns:
        Formatted prompt string
    """
    # Template based on original single_cell_cell_type_conditional_generation_prompts.json
    prompt = (
        f"Generate a {organism} {cell_type} single cell as a list of "
        f"{n_genes} genes ranked by expression.\n\n"
        f"Genes:"
    )
    return prompt


# Optional: convenience function for batch formatting
def format_batch_cell_type_prediction(
    cell_sentences: list[str],
    n_genes: int = 200,
    organism: str = 'Homo sapiens'
) -> list[str]:
    """Format multiple cells for cell type prediction."""
    return [
        format_cell_type_prediction(s, n_genes, organism)
        for s in cell_sentences
    ]


def format_batch_cell_generation(
    cell_types: list[str],
    n_genes: int = 200,
    organism: str = 'Homo sapiens'
) -> list[str]:
    """Format multiple cell types for generation."""
    return [
        format_cell_generation(ct, n_genes, organism)
        for ct in cell_types
    ]
