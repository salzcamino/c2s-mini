# c2s-mini API Reference

Complete API documentation for c2s-mini v0.1.0.

## Table of Contents

- [Core Classes](#core-classes)
  - [C2SData](#c2sdata)
  - [C2SModel](#c2smodel)
- [High-Level Functions](#high-level-functions)
  - [predict_cell_types](#predict_cell_types)
  - [generate_cells](#generate_cells)
  - [embed_cells](#embed_cells)
- [Utility Functions](#utility-functions)
  - [generate_vocabulary](#generate_vocabulary)
  - [generate_sentences](#generate_sentences)
  - [post_process_generated_cell_sentences](#post_process_generated_cell_sentences)
  - [reconstruct_expression_from_cell_sentence](#reconstruct_expression_from_cell_sentence)
- [Prompt Formatting](#prompt-formatting)
  - [format_cell_type_prediction](#format_cell_type_prediction)
  - [format_cell_generation](#format_cell_generation)
  - [truncate_sentence](#truncate_sentence)

---

## Core Classes

### C2SData

Lightweight wrapper for cell sentence data.

```python
from c2s_mini import C2SData
```

#### Constructor

```python
C2SData(
    vocab: OrderedDict,
    sentences: list[str],
    metadata: Optional[pd.DataFrame] = None
)
```

**Parameters:**
- `vocab` (OrderedDict): Gene vocabulary mapping gene names (uppercase) → expression counts
- `sentences` (list[str]): List of cell sentence strings
- `metadata` (Optional[pd.DataFrame]): Cell metadata (one row per cell)

**Raises:**
- `ValueError`: If metadata length doesn't match number of sentences

**Example:**
```python
from collections import OrderedDict
import pandas as pd

vocab = OrderedDict([('GENE1', 100), ('GENE2', 50)])
sentences = ['GENE1 GENE2', 'GENE2 GENE1']
metadata = pd.DataFrame({'cell_type': ['T cell', 'B cell']})

csdata = C2SData(vocab=vocab, sentences=sentences, metadata=metadata)
```

#### Class Methods

##### from_anndata

```python
@classmethod
C2SData.from_anndata(
    adata,
    delimiter: str = ' ',
    random_state: int = 42,
    include_obs_columns: Optional[list[str]] = None
) -> C2SData
```

Create C2SData from AnnData object.

**Parameters:**
- `adata` (AnnData): AnnData object (obs=cells, vars=genes)
- `delimiter` (str): Gene separator in sentences. Default: `' '`
- `random_state` (int): Random seed for tie-breaking. Default: `42`
- `include_obs_columns` (Optional[list[str]]): Column names from `adata.obs` to include in metadata

**Returns:**
- `C2SData`: New C2SData object

**Raises:**
- `ValueError`: If requested metadata columns don't exist in `adata.obs`

**Warnings:**
- Warns if `adata.var_names` contains Ensembl IDs instead of gene names
- Warns if more variables than observations (possible transpose needed)

**Example:**
```python
import scanpy as sc

adata = sc.datasets.pbmc3k()
csdata = C2SData.from_anndata(
    adata,
    include_obs_columns=['cell_type', 'batch']
)
```

#### Instance Methods

##### get_sentences

```python
def get_sentences() -> list[str]
```

Return list of cell sentences.

**Returns:**
- `list[str]`: List of cell sentence strings

**Example:**
```python
sentences = csdata.get_sentences()
print(f"First sentence: {sentences[0][:100]}...")
```

##### get_vocab

```python
def get_vocab() -> OrderedDict
```

Return vocabulary dictionary.

**Returns:**
- `OrderedDict`: Gene names (uppercase) → expression counts

**Example:**
```python
vocab = csdata.get_vocab()
print(f"Vocabulary size: {len(vocab)}")
print(f"Top genes: {list(vocab.keys())[:10]}")
```

##### to_dict

```python
def to_dict() -> dict
```

Convert to dictionary format.

**Returns:**
- `dict`: Dictionary with keys:
  - `'sentences'`: list of cell sentences
  - `'vocab'`: OrderedDict of vocabulary
  - `'metadata'`: DataFrame of metadata (or None)

**Example:**
```python
data_dict = csdata.to_dict()
# Save to JSON
import json
json_dict = {
    'sentences': data_dict['sentences'],
    'vocab': dict(data_dict['vocab'])
}
with open('cells.json', 'w') as f:
    json.dump(json_dict, f)
```

#### Special Methods

##### \_\_len\_\_

```python
def __len__() -> int
```

Return number of cells.

**Example:**
```python
print(f"Dataset has {len(csdata)} cells")
```

##### \_\_str\_\_ / \_\_repr\_\_

```python
def __str__() -> str
```

Human-readable summary.

**Example:**
```python
print(csdata)  # C2SData(n_cells=2700, n_genes=1838, metadata_cols=2)
```

#### Attributes

- `vocab` (OrderedDict): Gene vocabulary
- `sentences` (list[str]): Cell sentences
- `metadata` (Optional[pd.DataFrame]): Cell metadata

---

### C2SModel

Wrapper for Cell2Sentence model inference.

```python
from c2s_mini import C2SModel
```

#### Constructor

```python
C2SModel(device: str = 'auto')
```

Load pythia-160m-c2s model from HuggingFace.

**Parameters:**
- `device` (str): Device for inference. Options:
  - `'auto'`: Auto-detect (CUDA if available, else CPU)
  - `'cuda'`: Force CUDA (raises error if unavailable)
  - `'cpu'`: Force CPU

**Raises:**
- `RuntimeError`: If CUDA requested but not available

**Attributes:**
- `device` (str): Device being used ('cuda' or 'cpu')
- `model`: HuggingFace model instance
- `tokenizer`: HuggingFace tokenizer instance

**Example:**
```python
# Auto-detect device
model = C2SModel()

# Force CPU
model = C2SModel(device='cpu')

# Force CUDA
model = C2SModel(device='cuda')
```

#### Instance Methods

##### generate_from_prompt

```python
def generate_from_prompt(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    **kwargs
) -> str
```

Generate text from a single prompt.

**Parameters:**
- `prompt` (str): Input text prompt
- `max_tokens` (int): Maximum tokens to generate. Default: `1024`
- `temperature` (float): Sampling temperature. Default: `1.0`
- `top_p` (float): Nucleus sampling parameter. Default: `1.0`
- `do_sample` (bool): Whether to use sampling. Default: `True`
- `**kwargs`: Additional parameters passed to `model.generate()`

**Returns:**
- `str`: Generated text (excluding input prompt)

**Example:**
```python
prompt = "Predict the cell type: CD3D CD3E CD8A"
result = model.generate_from_prompt(
    prompt,
    max_tokens=50,
    temperature=0.8
)
print(f"Prediction: {result}")
```

##### generate_batch

```python
def generate_batch(
    prompts: list[str],
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    **kwargs
) -> list[str]
```

Generate text from multiple prompts (batched).

**Parameters:**
- `prompts` (list[str]): List of input prompts
- `max_tokens` (int): Maximum tokens per prompt. Default: `1024`
- `temperature` (float): Sampling temperature. Default: `1.0`
- `top_p` (float): Nucleus sampling parameter. Default: `1.0`
- `do_sample` (bool): Whether to use sampling. Default: `True`
- `**kwargs`: Additional parameters passed to `model.generate()`

**Returns:**
- `list[str]`: Generated texts (excluding input prompts)

**Example:**
```python
prompts = [
    "Predict: CD3D CD3E",
    "Predict: CD19 MS4A1",
    "Predict: NKG7 GZMA"
]
results = model.generate_batch(prompts, max_tokens=30)
for i, result in enumerate(results):
    print(f"Cell {i}: {result}")
```

##### embed_cell

```python
def embed_cell(prompt: str) -> np.ndarray
```

Get embedding for a single cell.

**Parameters:**
- `prompt` (str): Formatted cell sentence prompt

**Returns:**
- `np.ndarray`: Embedding vector (shape: `[hidden_size]`)

**Example:**
```python
prompt = "CD3D CD3E CD8A CD8B"
embedding = model.embed_cell(prompt)
print(f"Embedding shape: {embedding.shape}")  # (768,)
```

##### embed_batch

```python
def embed_batch(prompts: list[str]) -> np.ndarray
```

Get embeddings for multiple cells (batched).

**Parameters:**
- `prompts` (list[str]): List of formatted prompts

**Returns:**
- `np.ndarray`: Embeddings array (shape: `[n_cells, hidden_size]`)

**Example:**
```python
prompts = ["CD3D CD3E", "CD19 MS4A1", "NKG7 GZMA"]
embeddings = model.embed_batch(prompts)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 768)

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)
```

#### Special Methods

##### \_\_repr\_\_

```python
def __repr__() -> str
```

String representation.

**Example:**
```python
print(model)  # C2SModel(model=vandijklab/pythia-160m-c2s, device=cuda)
```

---

## High-Level Functions

### predict_cell_types

```python
from c2s_mini import predict_cell_types

def predict_cell_types(
    csdata: C2SData,
    model: C2SModel,
    n_genes: int = 200,
    organism: str = 'Homo sapiens',
    batch_size: int = 8,
    **generation_kwargs
) -> list[str]
```

Predict cell types for all cells in dataset.

**Parameters:**
- `csdata` (C2SData): Dataset with cell sentences
- `model` (C2SModel): Model for inference
- `n_genes` (int): Number of top genes to use. Default: `200`
- `organism` (str): Organism name. Default: `'Homo sapiens'`
- `batch_size` (int): Batch size for inference. Default: `8`
- `**generation_kwargs`: Additional args for `model.generate_batch()` (e.g., `max_tokens`, `temperature`)

**Returns:**
- `list[str]`: Predicted cell type strings (one per cell)

**Example:**
```python
import scanpy as sc
from c2s_mini import C2SData, C2SModel, predict_cell_types

adata = sc.datasets.pbmc3k()
csdata = C2SData.from_anndata(adata)
model = C2SModel()

predictions = predict_cell_types(
    csdata,
    model,
    n_genes=100,
    batch_size=16,
    max_tokens=50,
    temperature=0.7
)

print(f"Predictions: {predictions[:10]}")
```

---

### generate_cells

```python
from c2s_mini import generate_cells

def generate_cells(
    cell_types: list[str],
    model: C2SModel,
    n_genes: int = 200,
    organism: str = 'Homo sapiens',
    batch_size: int = 8,
    **generation_kwargs
) -> list[str]
```

Generate cell sentences conditioned on cell types.

**Parameters:**
- `cell_types` (list[str]): Cell type labels to generate
- `model` (C2SModel): Model for inference
- `n_genes` (int): Number of genes to generate. Default: `200`
- `organism` (str): Organism name. Default: `'Homo sapiens'`
- `batch_size` (int): Batch size for inference. Default: `8`
- `**generation_kwargs`: Additional args for `model.generate_batch()`

**Returns:**
- `list[str]`: Generated cell sentence strings

**Example:**
```python
from c2s_mini import C2SModel, generate_cells

model = C2SModel()
cell_types = ['T cell', 'B cell', 'NK cell', 'Monocyte']

generated = generate_cells(
    cell_types,
    model,
    n_genes=150,
    max_tokens=512,
    temperature=1.0
)

for cell_type, sentence in zip(cell_types, generated):
    genes = sentence.split()[:10]
    print(f"{cell_type}: {' '.join(genes)}...")
```

---

### embed_cells

```python
from c2s_mini import embed_cells

def embed_cells(
    csdata: C2SData,
    model: C2SModel,
    n_genes: int = 200,
    organism: str = 'Homo sapiens',
    batch_size: int = 8
) -> np.ndarray
```

Generate embeddings for all cells in dataset.

**Parameters:**
- `csdata` (C2SData): Dataset with cell sentences
- `model` (C2SModel): Model for inference
- `n_genes` (int): Number of genes to use. Default: `200`
- `organism` (str): Organism name. Default: `'Homo sapiens'`
- `batch_size` (int): Batch size for inference. Default: `8`

**Returns:**
- `np.ndarray`: Embeddings array (shape: `[n_cells, embedding_dim]`)

**Example:**
```python
import scanpy as sc
from c2s_mini import C2SData, C2SModel, embed_cells

adata = sc.datasets.pbmc3k()
csdata = C2SData.from_anndata(adata)
model = C2SModel()

embeddings = embed_cells(csdata, model, n_genes=100, batch_size=16)
print(f"Shape: {embeddings.shape}")  # (2700, 768)

# Use for UMAP
import umap
reducer = umap.UMAP()
umap_coords = reducer.fit_transform(embeddings)

# Plot
import matplotlib.pyplot as plt
plt.scatter(umap_coords[:, 0], umap_coords[:, 1])
plt.show()
```

---

## Utility Functions

### generate_vocabulary

```python
from c2s_mini.utils import generate_vocabulary

def generate_vocabulary(adata) -> OrderedDict
```

Create vocabulary from AnnData object.

**Parameters:**
- `adata` (AnnData): AnnData object (obs=cells, vars=genes)

**Returns:**
- `OrderedDict`: Gene names (uppercase) → non-zero cell counts

**Warnings:**
- Warns if more variables than observations

**Example:**
```python
import scanpy as sc
from c2s_mini.utils import generate_vocabulary

adata = sc.datasets.pbmc3k()
vocab = generate_vocabulary(adata)

print(f"Vocabulary size: {len(vocab)}")
for gene, count in list(vocab.items())[:10]:
    print(f"{gene}: {count}")
```

---

### generate_sentences

```python
from c2s_mini.utils import generate_sentences

def generate_sentences(
    adata,
    vocab: OrderedDict,
    delimiter: str = ' ',
    random_state: int = 42
) -> list[str]
```

Transform expression matrix to cell sentences.

**Parameters:**
- `adata` (AnnData): AnnData object
- `vocab` (OrderedDict): Vocabulary from `generate_vocabulary()`
- `delimiter` (str): Gene separator. Default: `' '`
- `random_state` (int): Random seed for tie-breaking. Default: `42`

**Returns:**
- `list[str]`: Cell sentence strings

**Example:**
```python
import scanpy as sc
from c2s_mini.utils import generate_vocabulary, generate_sentences

adata = sc.datasets.pbmc3k()
vocab = generate_vocabulary(adata)
sentences = generate_sentences(adata, vocab)

print(f"Generated {len(sentences)} sentences")
print(f"First sentence: {sentences[0][:200]}...")
```

---

### post_process_generated_cell_sentences

```python
from c2s_mini.utils import post_process_generated_cell_sentences

def post_process_generated_cell_sentences(
    cell_sentence: str,
    vocab_list: list,
    replace_nonsense_string: str = "NOT_A_GENE"
) -> tuple[list, int]
```

Clean generated sentences.

**Parameters:**
- `cell_sentence` (str): Generated cell sentence
- `vocab_list` (list): Valid gene names (uppercase)
- `replace_nonsense_string` (str): Placeholder for invalid genes. Default: `"NOT_A_GENE"`

**Returns:**
- `tuple[list, int]`:
  - Cleaned gene list
  - Number of genes replaced

**Example:**
```python
from c2s_mini.utils import post_process_generated_cell_sentences

vocab = ["CD3D", "CD3E", "CD8A"]
sentence = "CD3D INVALID CD3E CD3D"

cleaned, n_replaced = post_process_generated_cell_sentences(sentence, vocab)
print(f"Cleaned: {cleaned}")  # CD3D appears once at avg position
print(f"Replaced {n_replaced} invalid genes")
```

---

### reconstruct_expression_from_cell_sentence

```python
from c2s_mini.utils import reconstruct_expression_from_cell_sentence

def reconstruct_expression_from_cell_sentence(
    cell_sentence_str: str,
    delimiter: str,
    vocab_list: list,
    slope: float,
    intercept: float
) -> np.ndarray
```

Reconstruct expression vector from cell sentence.

**Parameters:**
- `cell_sentence_str` (str): Cell sentence string
- `delimiter` (str): Gene separator
- `vocab_list` (list): All gene names (defines order)
- `slope` (float): Linear model slope (log rank → log expression)
- `intercept` (float): Linear model intercept

**Returns:**
- `np.ndarray`: Expression vector (length = len(vocab_list))

**Example:**
```python
from c2s_mini.utils import reconstruct_expression_from_cell_sentence

vocab = ["GENE1", "GENE2", "GENE3", "GENE4"]
sentence = "GENE3 GENE1 GENE4"

expr = reconstruct_expression_from_cell_sentence(
    sentence,
    delimiter=' ',
    vocab_list=vocab,
    slope=-1.0,
    intercept=5.0
)

print(f"Expression: {expr}")
# GENE3 has highest, GENE2 has 0
```

---

## Prompt Formatting

### format_cell_type_prediction

```python
from c2s_mini.prompts import format_cell_type_prediction

def format_cell_type_prediction(
    cell_sentence: str,
    n_genes: int = 200,
    organism: str = 'Homo sapiens'
) -> str
```

Format prompt for cell type prediction.

**Parameters:**
- `cell_sentence` (str): Space-separated gene names
- `n_genes` (int): Number of genes to use. Default: `200`
- `organism` (str): Organism name. Default: `'Homo sapiens'`

**Returns:**
- `str`: Formatted prompt

**Example:**
```python
from c2s_mini.prompts import format_cell_type_prediction

sentence = "CD3D CD3E CD8A CD8B CCL5 GZMA"
prompt = format_cell_type_prediction(sentence, n_genes=5)
print(prompt)
```

---

### format_cell_generation

```python
from c2s_mini.prompts import format_cell_generation

def format_cell_generation(
    cell_type: str,
    n_genes: int = 200,
    organism: str = 'Homo sapiens'
) -> str
```

Format prompt for cell generation.

**Parameters:**
- `cell_type` (str): Target cell type
- `n_genes` (int): Number of genes to generate. Default: `200`
- `organism` (str): Organism name. Default: `'Homo sapiens'`

**Returns:**
- `str`: Formatted prompt

**Example:**
```python
from c2s_mini.prompts import format_cell_generation

prompt = format_cell_generation("B cell", n_genes=150)
print(prompt)
```

---

### truncate_sentence

```python
from c2s_mini.prompts import truncate_sentence

def truncate_sentence(
    sentence: str,
    n_genes: int,
    delimiter: str = ' '
) -> str
```

Truncate cell sentence to first N genes.

**Parameters:**
- `sentence` (str): Full cell sentence
- `n_genes` (int): Number of genes to keep
- `delimiter` (str): Gene separator. Default: `' '`

**Returns:**
- `str`: Truncated sentence

**Example:**
```python
from c2s_mini.prompts import truncate_sentence

sentence = "GENE1 GENE2 GENE3 GENE4 GENE5"
truncated = truncate_sentence(sentence, n_genes=3)
print(truncated)  # "GENE1 GENE2 GENE3"
```

---

## Type Aliases

```python
from typing import OrderedDict
from collections import OrderedDict
import pandas as pd
import numpy as np

# Common type aliases used throughout the API
Vocabulary = OrderedDict[str, int]  # Gene name → count
CellSentence = str  # Space-separated gene names
CellSentences = list[str]  # Multiple cell sentences
Embedding = np.ndarray  # Single embedding vector
Embeddings = np.ndarray  # Multiple embeddings (2D array)
```

---

## Constants

```python
# Model name
MODEL_NAME = "vandijklab/pythia-160m-c2s"

# Default parameters
DEFAULT_N_GENES = 200
DEFAULT_ORGANISM = "Homo sapiens"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_DELIMITER = ' '
DEFAULT_RANDOM_STATE = 42
```

---

## Error Reference

### ValueError

Raised when:
- Metadata length doesn't match sentence count in C2SData
- Requested metadata columns don't exist in AnnData.obs

### RuntimeError

Raised when:
- CUDA requested but not available

---

## Best Practices

### Device Management

```python
# Automatic device selection (recommended)
model = C2SModel()

# Check device
print(f"Using device: {model.device}")

# Clear CUDA cache if needed
if model.device == 'cuda':
    import torch
    torch.cuda.empty_cache()
```

### Memory Efficiency

```python
# Use appropriate batch sizes
if model.device == 'cuda':
    batch_size = 32  # Larger for GPU
else:
    batch_size = 4   # Smaller for CPU

# Process large datasets in chunks
n_cells = len(csdata)
chunk_size = 1000

all_predictions = []
for i in range(0, n_cells, chunk_size):
    chunk = C2SData(
        vocab=csdata.vocab,
        sentences=csdata.sentences[i:i+chunk_size],
        metadata=csdata.metadata[i:i+chunk_size] if csdata.metadata is not None else None
    )
    predictions = predict_cell_types(chunk, model, batch_size=batch_size)
    all_predictions.extend(predictions)
```

### Error Handling

```python
try:
    csdata = C2SData.from_anndata(
        adata,
        include_obs_columns=['cell_type']
    )
except ValueError as e:
    print(f"Metadata error: {e}")
    # Fall back to no metadata
    csdata = C2SData.from_anndata(adata)
```

---

## Version Information

```python
import c2s_mini
print(c2s_mini.__version__)  # 0.1.0
```
