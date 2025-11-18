"""
Core data transformation utilities for c2s-mini.

This module provides functions to transform single-cell RNA sequencing data
into "cell sentences" (space-separated gene names ordered by descending expression)
and vice versa.

Reference:
    Original implementation: https://github.com/vandijklab/cell2sentence
"""

import sys
from collections import OrderedDict, Counter

import numpy as np
from scipy import sparse
from sklearn.utils import shuffle
from tqdm import tqdm


def generate_vocabulary(adata):
    """
    Create a vocabulary dictionary where each key represents a single gene
    token and the value represents the number of non-zero cells in the provided
    count matrix.

    Args:
        adata: AnnData object to generate vocabulary from. Expects that
            `obs` correspond to cells and `vars` correspond to genes.

    Returns:
        OrderedDict: Ordered dictionary mapping gene names (uppercase) to
            non-zero cell counts.

    Example:
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc3k()
        >>> vocab = generate_vocabulary(adata)
        >>> print(f"Vocabulary size: {len(vocab)}")
    """
    if len(adata.var) > len(adata.obs):
        print(
            (
                "WARN: more variables ({}) than observations ({})... "
                + "did you mean to transpose the object (e.g. adata.T)?"
            ).format(len(adata.var), len(adata.obs)),
            file=sys.stderr,
        )

    vocabulary = OrderedDict()
    gene_sums = np.ravel(np.sum(adata.X > 0, axis=0))

    for i, name in enumerate(adata.var_names):
        vocabulary[name.upper()] = gene_sums[i]  # keys are all uppercase gene names

    return vocabulary


def generate_sentences(adata, vocab, delimiter=' ', random_state=42):
    """
    Transform expression matrix to cell sentences.

    Sentences contain gene "words" denoting genes with non-zero expression.
    Genes are ordered from highest expression to lowest expression.

    Args:
        adata: AnnData object to generate cell sentences from. Expects that
            `obs` correspond to cells and `vars` correspond to genes.
        vocab: OrderedDict with gene names as keys (from generate_vocabulary).
        delimiter: String separator for gene names (default: ' ').
        random_state: Random seed for tie-breaking in gene ranking (default: 42).

    Returns:
        list: List of cell sentence strings, one per cell.

    Example:
        >>> import scanpy as sc
        >>> from c2s_mini.utils import generate_vocabulary, generate_sentences
        >>> adata = sc.datasets.pbmc3k()
        >>> vocab = generate_vocabulary(adata)
        >>> sentences = generate_sentences(adata, vocab)
        >>> print(sentences[0][:100])  # First 100 chars of first sentence
    """
    np.random.seed(random_state)

    if len(adata.var) > len(adata.obs):
        print(
            (
                "WARN: more variables ({}) than observations ({}), "
                + "did you mean to transpose the object (e.g. adata.T)?"
            ).format(len(adata.var), len(adata.obs)),
            file=sys.stderr,
        )

    mat = sparse.csr_matrix(adata.X)
    enc_map = list(vocab.keys())

    sentences = []
    for i in tqdm(range(mat.shape[0]), desc="Generating cell sentences"):
        # For row i, [indptr[i]:indptr[i+1]] returns the indices of elements
        # to take from data and indices corresponding to row i
        cols = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        vals = mat.data[mat.indptr[i] : mat.indptr[i + 1]]
        cols, vals = shuffle(cols, vals, random_state=random_state)
        # Sort by value (descending) and join gene names
        sentence = delimiter.join([
            enc_map[x] for x in cols[np.argsort(-vals, kind="stable")]
        ])
        sentences.append(sentence)

    return sentences


def post_process_generated_cell_sentences(
    cell_sentence: str,
    vocab_list: list,
    replace_nonsense_string: str = "NOT_A_GENE",
):
    """
    Clean generated sentences by replacing non-gene words and averaging
    duplicate gene positions.

    This function processes LLM-generated cell sentences by:
    1. Replacing words not in the vocabulary with a placeholder
    2. Handling duplicate genes by averaging their positions

    Args:
        cell_sentence: String representing a generated cell sentence.
        vocab_list: List of all valid gene names (uppercase).
        replace_nonsense_string: Placeholder for invalid genes (default: "NOT_A_GENE").
            Should not match any real gene name.

    Returns:
        tuple: (post_processed_gene_list, num_genes_replaced)
            - post_processed_gene_list: List of cleaned gene names
            - num_genes_replaced: Count of non-gene words replaced

    Example:
        >>> vocab = ["GENE1", "GENE2", "GENE3"]
        >>> sentence = "GENE1 INVALID GENE2 GENE1"
        >>> cleaned, n_replaced = post_process_generated_cell_sentences(sentence, vocab)
        >>> print(f"Replaced {n_replaced} invalid genes")
        >>> print(f"Cleaned: {cleaned}")  # GENE1 appears once at avg position
    """
    # Convert the cell sentence to uppercase and split into words
    words = cell_sentence.upper().split(" ")

    # Replace words not in the vocabulary with the replace_nonsense_string
    generated_gene_names = [
        word if word in vocab_list else replace_nonsense_string
        for word in words
    ]
    num_genes_replaced = generated_gene_names.count(replace_nonsense_string)

    # Calculate average ranks for duplicate genes
    gene_name_to_occurrences = Counter(generated_gene_names)
    post_processed_sentence = generated_gene_names.copy()

    for gene_name in gene_name_to_occurrences:
        if (gene_name_to_occurrences[gene_name] > 1
            and gene_name != replace_nonsense_string):
            # Find positions of all occurrences of duplicated gene
            # Note: using post_processed_sentence here; since duplicates are being
            # removed, the list will get shorter. Getting indices in original list
            # would no longer give accurate positions.
            occurrence_positions = [
                idx for idx, elem in enumerate(post_processed_sentence)
                if elem == gene_name
            ]
            average_position = int(sum(occurrence_positions) / len(occurrence_positions))

            # Remove all occurrences
            post_processed_sentence = [
                elem for elem in post_processed_sentence if elem != gene_name
            ]
            # Reinsert gene at average position
            post_processed_sentence.insert(average_position, gene_name)

    return post_processed_sentence, num_genes_replaced


def reconstruct_expression_from_cell_sentence(
    cell_sentence_str: str,
    delimiter: str,
    vocab_list: list,
    slope: float,
    intercept: float,
):
    """
    Reconstruct an expression vector from a cell sentence.

    This function converts a ranked list of genes back into an expression
    vector using a linear model that maps log(rank) to expression values.

    Args:
        cell_sentence_str: String representing a cell sentence (space-separated genes).
        delimiter: Character which separates gene names in the cell sentence.
        vocab_list: List of all gene feature names. The output expression vector
            will be ordered following this list.
        slope: Slope of linear model fit on log(rank) vs log(expression).
        intercept: Intercept of linear model fit on log(rank) vs log(expression).

    Returns:
        np.ndarray: Expression vector with same length as vocab_list.
            Genes not in the sentence have expression value 0.

    Example:
        >>> vocab = ["GENE1", "GENE2", "GENE3", "GENE4"]
        >>> sentence = "GENE3 GENE1 GENE4"
        >>> expr = reconstruct_expression_from_cell_sentence(
        ...     sentence, delimiter=' ', vocab_list=vocab,
        ...     slope=-1.0, intercept=5.0
        ... )
        >>> print(expr)  # GENE3 has highest expr, GENE2 has 0
    """
    # Split cell sentence string into list of gene "words"
    cell_sentence = cell_sentence_str.split(delimiter)

    # Create a mapping from gene names to their vocab indices for O(1) lookups
    gene_to_index = {gene: idx for idx, gene in enumerate(vocab_list)}

    # Initialize the expression vector with zeros
    expression_vector = np.zeros(len(vocab_list), dtype=np.float32)

    # Pre-compute the log rank values for all positions in cell sentence
    log_ranks = np.log10(1 + np.arange(len(cell_sentence)))

    # Calculate gene expression values and update the expression vector
    for pos, gene_name in enumerate(cell_sentence):
        gene_idx_in_vector = gene_to_index.get(gene_name)
        if gene_idx_in_vector is not None:  # gene is in vocab_list
            gene_expr_val = intercept + (slope * log_ranks[pos])
            expression_vector[gene_idx_in_vector] = gene_expr_val

    return expression_vector
