# c2s-mini

**Miniature implementation of Cell2Sentence for single-cell analysis with LLMs**

A lightweight, inference-focused implementation of [Cell2Sentence](https://github.com/vandijklab/cell2sentence) that transforms single-cell RNA sequencing data into "cell sentences" - space-separated gene names ordered by descending expression - for analysis with Large Language Models.

## Features

- **Simple API**: Easy-to-use functions for common single-cell tasks
- **Lightweight**: ~400-600 lines of core functionality (vs. ~1,500 in full implementation)
- **Inference-focused**: Cell type prediction, generation, and embedding
- **Model**: Supports `pythia-160m-c2s` (160M parameter model)
- **Compatible**: Works with AnnData objects from scanpy

## Installation

### From source (development)

```bash
git clone https://github.com/salzcamino/c2s-mini.git
cd c2s-mini
pip install -e .
```

### With development dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import scanpy as sc
from c2s_mini import C2SData, C2SModel, predict_cell_types

# Load your single-cell data
adata = sc.datasets.pbmc3k()

# Convert to Cell2Sentence format
csdata = C2SData.from_anndata(adata)

# Load the model
model = C2SModel(device='auto')  # Uses CUDA if available

# Predict cell types
predictions = predict_cell_types(
    csdata,
    model,
    n_genes=100,
    batch_size=8
)

print(f"Predicted {len(predictions)} cell types")
```

## What is Cell2Sentence?

Cell2Sentence transforms gene expression data into natural language by:

1. **Ranking genes** by expression level in each cell
2. **Creating "cell sentences"** - space-separated gene names in descending order
3. **Using LLMs** to analyze these sentences for biological insights

**Example cell sentence:**
```
CD3D CD3E CD8A CD8B GZMK CCL5 NKG7 CST7 GZMA PRF1 ...
```

This representation allows pre-trained language models to:
- Predict cell types
- Generate synthetic cells
- Create cell embeddings
- Answer biological questions

## Core Components

### Data Transformation
```python
from c2s_mini import C2SData

# Create from AnnData
csdata = C2SData.from_anndata(adata, include_obs_columns=['cell_type'])

# Access sentences and vocabulary
sentences = csdata.get_sentences()
vocab = csdata.get_vocab()
```

### Model Inference
```python
from c2s_mini import C2SModel

# Load model (pythia-160m-c2s)
model = C2SModel(device='cuda')

# Generate from prompt
prompt = "Predict the cell type: CD3D CD3E CD8A"
result = model.generate_from_prompt(prompt, max_tokens=50)
```

### High-Level Tasks
```python
from c2s_mini import predict_cell_types, generate_cells, embed_cells

# Cell type prediction
predictions = predict_cell_types(csdata, model, n_genes=200)

# Cell generation
cell_types = ['T cell', 'B cell', 'NK cell']
generated = generate_cells(cell_types, model, n_genes=200)

# Cell embeddings
embeddings = embed_cells(csdata, model, n_genes=200)
```

## API Reference

### C2SData

**Class for managing cell sentence data**

```python
# Create from AnnData
csdata = C2SData.from_anndata(
    adata,
    delimiter=' ',                    # Gene separator (default: ' ')
    random_state=42,                  # Random seed for tie-breaking
    include_obs_columns=['cell_type'] # Metadata columns to preserve
)

# Access data
sentences = csdata.get_sentences()    # List of cell sentences
vocab = csdata.get_vocab()            # OrderedDict of gene names → counts
metadata = csdata.metadata            # DataFrame with cell metadata

# Properties
len(csdata)                           # Number of cells
str(csdata)                           # Human-readable summary
```

### C2SModel

**Wrapper for the pythia-160m-c2s model**

```python
# Load model
model = C2SModel(device='auto')  # 'auto', 'cuda', or 'cpu'

# Single generation
result = model.generate_from_prompt(
    prompt="Predict the cell type: CD3D CD3E",
    max_tokens=50,
    temperature=1.0,
    top_p=1.0,
    do_sample=True
)

# Batch generation
results = model.generate_batch(
    prompts=["prompt1", "prompt2"],
    max_tokens=50
)

# Single embedding
embedding = model.embed_cell("CD3D CD3E CD8A")  # Returns np.ndarray

# Batch embeddings
embeddings = model.embed_batch(["cell1", "cell2"])  # Returns (n_cells, hidden_dim)
```

### High-Level Functions

#### predict_cell_types()
```python
predictions = predict_cell_types(
    csdata,                    # C2SData object
    model,                     # C2SModel instance
    n_genes=200,              # Number of top genes to use
    organism='Homo sapiens',  # 'Homo sapiens' or 'Mus musculus'
    batch_size=8,             # Batch size for inference
    max_tokens=50             # Max tokens to generate
)
# Returns: List[str] of predicted cell types
```

#### generate_cells()
```python
generated = generate_cells(
    cell_types=['T cell', 'B cell'],  # Cell types to generate
    model,                             # C2SModel instance
    n_genes=200,                      # Number of genes to generate
    organism='Homo sapiens',
    batch_size=8,
    max_tokens=512
)
# Returns: List[str] of generated cell sentences
```

#### embed_cells()
```python
embeddings = embed_cells(
    csdata,                    # C2SData object
    model,                     # C2SModel instance
    n_genes=200,              # Number of top genes to use
    organism='Homo sapiens',
    batch_size=8
)
# Returns: np.ndarray of shape (n_cells, embedding_dim)
```

### Prompt Formatting

```python
from c2s_mini.prompts import (
    format_cell_type_prediction,
    format_cell_generation,
    truncate_sentence
)

# Format for cell type prediction
prompt = format_cell_type_prediction(
    cell_sentence="CD3D CD3E CD8A ...",
    n_genes=100,
    organism='Homo sapiens'
)

# Format for cell generation
prompt = format_cell_generation(
    cell_type='T cell',
    n_genes=200,
    organism='Homo sapiens'
)

# Truncate sentence
truncated = truncate_sentence(
    sentence="GENE1 GENE2 GENE3 ...",
    n_genes=50,
    delimiter=' '
)
```

## Examples

See the `examples/` directory for Jupyter notebooks:

- `basic_usage.ipynb` - Data transformation and prompt formatting
- `cell_type_prediction.ipynb` - Full prediction pipeline

## Implementation Status

**Current Phase: 6 (Documentation & Examples)**

Completed phases:
- ✅ **Phase 0**: Project setup and structure
- ✅ **Phase 1**: Core transformation utilities (`utils.py`)
- ✅ **Phase 2**: Data wrapper class (`data.py`)
- ✅ **Phase 3**: Model wrapper class (`model.py`)
- ✅ **Phase 4**: Prompt formatting (`prompts.py`)
- ✅ **Phase 5**: High-level task functions (`tasks.py`)
- 🚧 **Phase 6**: Example notebooks and documentation
- ⏳ **Phase 7**: Test suite

All core functionality is implemented and ready to use! See [CLAUDE.md](CLAUDE.md) for detailed implementation roadmap.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- AnnData 0.8+
- Scanpy 1.9+

See [requirements.txt](requirements.txt) for complete list.

## Differences from Full Cell2Sentence

**c2s-mini** is a simplified version focused on inference:

| Feature | Full C2S | c2s-mini |
|---------|----------|----------|
| Model support | Multiple (GPT-2, Pythia, Gemma) | pythia-160m-c2s only |
| Training | ✅ Full fine-tuning | ❌ Inference only |
| Data backend | Arrow datasets | Simple dict/list |
| Multi-cell tasks | ✅ Tissue prediction, NLI | ❌ Single-cell only |
| Benchmarking | ✅ Plots and metrics | ❌ Not included |
| Code size | ~1,500 lines | ~400-600 lines |

## Citation

If you use c2s-mini or Cell2Sentence in your work, please cite:

```bibtex
@article{rizvi2024cell2sentence,
  title={Cell2Sentence: Teaching Large Language Models the Language of Biology},
  author={Rizvi, Syed Asad and others},
  journal={bioRxiv},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

Apache 2.0 (matching original Cell2Sentence)

## Acknowledgments

Based on [Cell2Sentence](https://github.com/vandijklab/cell2sentence) by the van Dijk Lab at Yale.

## Links

- **Original Cell2Sentence**: https://github.com/vandijklab/cell2sentence
- **Paper**: https://www.biorxiv.org/content/10.1101/2023.09.11.557287
- **Model**: https://huggingface.co/vandijklab/pythia-160m-c2s
- **van Dijk Lab**: https://www.vandijklab.org/

## Contributing

This project is structured for implementation across multiple sessions. See [CLAUDE.md](CLAUDE.md) for phase-by-phase implementation guidance.
