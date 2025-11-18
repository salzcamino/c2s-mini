# c2s-mini Usage Guide

Comprehensive guide to using c2s-mini for single-cell analysis with LLMs.

## Table of Contents

- [Quick Start](#quick-start)
- [Loading and Preparing Data](#loading-and-preparing-data)
- [Cell Type Prediction](#cell-type-prediction)
- [Cell Generation](#cell-generation)
- [Cell Embeddings](#cell-embeddings)
- [Advanced Usage](#advanced-usage)
- [Performance Optimization](#performance-optimization)
- [Integration with Scanpy](#integration-with-scanpy)
- [Common Workflows](#common-workflows)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/salzcamino/c2s-mini.git
cd c2s-mini

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Example

```python
import scanpy as sc
from c2s_mini import C2SData, C2SModel, predict_cell_types

# 1. Load data
adata = sc.datasets.pbmc3k()

# 2. Convert to Cell2Sentence format
csdata = C2SData.from_anndata(adata)

# 3. Load model
model = C2SModel(device='auto')

# 4. Predict cell types
predictions = predict_cell_types(
    csdata,
    model,
    n_genes=100,
    batch_size=8
)

print(f"Predicted {len(predictions)} cell types")
print(f"Sample predictions: {predictions[:5]}")
```

---

## Loading and Preparing Data

### From AnnData

c2s-mini works directly with AnnData objects from scanpy:

```python
import scanpy as sc
from c2s_mini import C2SData

# Load from scanpy datasets
adata = sc.datasets.pbmc3k()

# Or read from file
adata = sc.read_h5ad('my_data.h5ad')

# Or read from 10x
adata = sc.read_10x_mtx('filtered_matrices/')

# Convert to C2SData
csdata = C2SData.from_anndata(adata)
```

### Preprocessing Recommendations

While c2s-mini works with raw counts, some preprocessing can improve results:

```python
import scanpy as sc

# Load data
adata = sc.read_h5ad('my_data.h5ad')

# Basic QC (optional)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Important: Use gene symbols, not Ensembl IDs
if 'ENS' in str(adata.var_names[0]):
    print("Warning: Using Ensembl IDs. Consider converting to gene symbols.")

# Convert to C2SData
csdata = C2SData.from_anndata(adata)
```

### Preserving Metadata

Include cell annotations in your C2SData object:

```python
# Include specific columns
csdata = C2SData.from_anndata(
    adata,
    include_obs_columns=['cell_type', 'batch', 'sample']
)

# Access metadata
print(csdata.metadata.head())

# Use metadata for validation
ground_truth = csdata.metadata['cell_type']
```

### Working with Different Organisms

c2s-mini supports both human and mouse data:

```python
# Human data
csdata_human = C2SData.from_anndata(adata_human)
predictions = predict_cell_types(
    csdata_human,
    model,
    organism='Homo sapiens'
)

# Mouse data
csdata_mouse = C2SData.from_anndata(adata_mouse)
predictions = predict_cell_types(
    csdata_mouse,
    model,
    organism='Mus musculus'
)
```

---

## Cell Type Prediction

### Basic Prediction

```python
from c2s_mini import C2SData, C2SModel, predict_cell_types

# Load model once
model = C2SModel(device='auto')

# Predict cell types
predictions = predict_cell_types(
    csdata,
    model,
    n_genes=200,        # Number of top genes to use
    batch_size=8,       # Batch size for inference
    max_tokens=50       # Maximum tokens to generate
)
```

### Tuning Parameters

#### Number of Genes

```python
# More genes = more context, but slower
predictions_detailed = predict_cell_types(
    csdata, model,
    n_genes=500,    # More genes
    max_tokens=50
)

# Fewer genes = faster, but less context
predictions_fast = predict_cell_types(
    csdata, model,
    n_genes=50,     # Fewer genes
    max_tokens=50
)
```

#### Temperature and Sampling

```python
# Deterministic (greedy)
predictions_greedy = predict_cell_types(
    csdata, model,
    n_genes=200,
    do_sample=False
)

# Creative sampling (higher temperature)
predictions_creative = predict_cell_types(
    csdata, model,
    n_genes=200,
    temperature=1.5,
    top_p=0.95
)

# Conservative sampling (lower temperature)
predictions_conservative = predict_cell_types(
    csdata, model,
    n_genes=200,
    temperature=0.5,
    top_p=0.9
)
```

### Validation

```python
import pandas as pd

# Compare with ground truth
results = pd.DataFrame({
    'predicted': predictions,
    'ground_truth': csdata.metadata['cell_type']
})

# Calculate accuracy metrics
from sklearn.metrics import classification_report

print(classification_report(
    results['ground_truth'],
    results['predicted']
))

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(
    results['ground_truth'],
    results['predicted']
)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

### Batch Processing Large Datasets

```python
# Process in chunks for large datasets
def predict_large_dataset(csdata, model, chunk_size=1000):
    all_predictions = []

    for i in range(0, len(csdata), chunk_size):
        # Create chunk
        chunk_data = C2SData(
            vocab=csdata.vocab,
            sentences=csdata.sentences[i:i+chunk_size],
            metadata=csdata.metadata[i:i+chunk_size] if csdata.metadata else None
        )

        # Predict
        chunk_predictions = predict_cell_types(
            chunk_data,
            model,
            n_genes=200,
            batch_size=16
        )

        all_predictions.extend(chunk_predictions)

        print(f"Processed {min(i+chunk_size, len(csdata))}/{len(csdata)} cells")

    return all_predictions

# Use it
predictions = predict_large_dataset(csdata, model, chunk_size=500)
```

---

## Cell Generation

### Basic Generation

```python
from c2s_mini import C2SModel, generate_cells

model = C2SModel()

# Define cell types to generate
cell_types = ['T cell', 'B cell', 'NK cell', 'Monocyte']

# Generate
generated = generate_cells(
    cell_types,
    model,
    n_genes=200,
    max_tokens=512
)

# Inspect results
for cell_type, sentence in zip(cell_types, generated):
    genes = sentence.split()[:20]
    print(f"\n{cell_type}:")
    print(' '.join(genes))
```

### Generating Multiple Cells per Type

```python
# Generate 5 B cells
cell_types = ['B cell'] * 5

generated = generate_cells(
    cell_types,
    model,
    n_genes=200,
    temperature=1.0  # Use sampling for variety
)

# Each will be slightly different due to sampling
for i, sentence in enumerate(generated):
    print(f"B cell #{i+1}: {sentence[:100]}...")
```

### Post-Processing Generated Cells

```python
from c2s_mini.utils import post_process_generated_cell_sentences

# Generate cells
generated = generate_cells(['T cell'], model, n_genes=200)

# Get vocabulary
vocab_list = list(csdata.get_vocab().keys())

# Clean generated sentences
cleaned_sentences = []
for sentence in generated:
    cleaned, n_replaced = post_process_generated_cell_sentences(
        sentence,
        vocab_list
    )
    cleaned_sentences.append(' '.join(cleaned))
    print(f"Replaced {n_replaced} non-gene words")
```

### Creating Synthetic Datasets

```python
import numpy as np
import pandas as pd

# Generate synthetic dataset
cell_types = ['T cell'] * 100 + ['B cell'] * 100 + ['NK cell'] * 50

generated_sentences = generate_cells(
    cell_types,
    model,
    n_genes=200,
    batch_size=16,
    temperature=1.0
)

# Create synthetic C2SData
synthetic_csdata = C2SData(
    vocab=csdata.vocab,  # Use real vocabulary
    sentences=generated_sentences,
    metadata=pd.DataFrame({'cell_type': cell_types})
)

# Use for downstream analysis
print(f"Created synthetic dataset with {len(synthetic_csdata)} cells")
```

---

## Cell Embeddings

### Basic Embedding

```python
from c2s_mini import C2SModel, embed_cells
import numpy as np

model = C2SModel()

# Generate embeddings
embeddings = embed_cells(
    csdata,
    model,
    n_genes=200,
    batch_size=16
)

print(f"Embedding shape: {embeddings.shape}")  # (n_cells, 768)
```

### Dimensionality Reduction

```python
# UMAP
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
umap_coords = reducer.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    umap_coords[:, 0],
    umap_coords[:, 1],
    c=csdata.metadata['cell_type'].astype('category').cat.codes,
    cmap='tab20',
    alpha=0.6
)
plt.colorbar(scatter)
plt.title('UMAP of Cell Embeddings')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()
```

```python
# t-SNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30)
tsne_coords = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], alpha=0.6)
plt.title('t-SNE of Cell Embeddings')
plt.show()
```

### Clustering

```python
from sklearn.cluster import KMeans
import scanpy as sc

# K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Add to AnnData for scanpy visualization
adata.obs['c2s_cluster'] = clusters.astype(str)
adata.obsm['X_c2s'] = embeddings

# Visualize with scanpy
sc.pp.neighbors(adata, use_rep='X_c2s')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['c2s_cluster', 'cell_type'])
```

### Similarity Search

```python
from sklearn.metrics.pairwise import cosine_similarity

# Find most similar cells
def find_similar_cells(query_idx, embeddings, top_k=5):
    # Compute similarities
    similarities = cosine_similarity(
        embeddings[query_idx:query_idx+1],
        embeddings
    )[0]

    # Get top k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return top_indices, similarities[top_indices]

# Example
query_cell = 0
similar_cells, scores = find_similar_cells(query_cell, embeddings, top_k=10)

print(f"Top 10 cells similar to cell {query_cell}:")
for idx, score in zip(similar_cells, scores):
    print(f"  Cell {idx}: similarity = {score:.3f}")
```

### Cell Type Cohesion Analysis

```python
# Analyze how well cell types cluster
from sklearn.metrics import silhouette_score

# Compute silhouette score
labels = csdata.metadata['cell_type'].astype('category').cat.codes
score = silhouette_score(embeddings, labels)

print(f"Silhouette score: {score:.3f}")

# Per-cluster analysis
from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(embeddings, labels)

# Average silhouette by cell type
import pandas as pd
sil_df = pd.DataFrame({
    'cell_type': csdata.metadata['cell_type'],
    'silhouette': silhouette_vals
})

print("\nAverage silhouette by cell type:")
print(sil_df.groupby('cell_type')['silhouette'].mean().sort_values(ascending=False))
```

---

## Advanced Usage

### Custom Prompts

```python
from c2s_mini.prompts import truncate_sentence

# Get cell sentence
sentence = csdata.get_sentences()[0]

# Custom prompt
custom_prompt = f"""
Analyze this cell based on gene expression:
{truncate_sentence(sentence, n_genes=100)}

Is this cell activated or quiescent?
Answer:"""

# Generate
result = model.generate_from_prompt(custom_prompt, max_tokens=30)
print(result)
```

### Multi-Organism Analysis

```python
# Human and mouse data
csdata_human = C2SData.from_anndata(adata_human)
csdata_mouse = C2SData.from_anndata(adata_mouse)

model = C2SModel()

# Predict separately
pred_human = predict_cell_types(csdata_human, model, organism='Homo sapiens')
pred_mouse = predict_cell_types(csdata_mouse, model, organism='Mus musculus')

# Compare
print(f"Human predictions: {set(pred_human)}")
print(f"Mouse predictions: {set(pred_mouse)}")
```

### Converting Back to Expression

```python
from c2s_mini.utils import reconstruct_expression_from_cell_sentence
import numpy as np

# Get a generated sentence
generated = generate_cells(['B cell'], model, n_genes=200)[0]

# Define linear model parameters (from original paper)
slope = -0.5
intercept = 3.0

# Reconstruct expression
vocab_list = list(csdata.get_vocab().keys())
expr = reconstruct_expression_from_cell_sentence(
    generated,
    delimiter=' ',
    vocab_list=vocab_list,
    slope=slope,
    intercept=intercept
)

# Create AnnData from reconstructed expression
import anndata

reconstructed_adata = anndata.AnnData(
    X=expr.reshape(1, -1),
    var=pd.DataFrame(index=vocab_list)
)

print(f"Reconstructed expression shape: {reconstructed_adata.shape}")
```

---

## Performance Optimization

### GPU Acceleration

```python
# Check if CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Use GPU
model = C2SModel(device='cuda')

# Increase batch size for GPU
predictions = predict_cell_types(
    csdata,
    model,
    batch_size=32 if model.device == 'cuda' else 8
)
```

### Memory Management

```python
import torch

# Clear CUDA cache between operations
if model.device == 'cuda':
    torch.cuda.empty_cache()

# Process in smaller chunks if running out of memory
chunk_size = 500  # Adjust based on available memory

for i in range(0, len(csdata), chunk_size):
    chunk = C2SData(
        vocab=csdata.vocab,
        sentences=csdata.sentences[i:i+chunk_size]
    )
    predictions = predict_cell_types(chunk, model)

    # Process predictions
    # ...

    # Clear cache
    if model.device == 'cuda':
        torch.cuda.empty_cache()
```

### Batch Size Tuning

```python
import time

# Test different batch sizes
batch_sizes = [4, 8, 16, 32, 64]
times = []

for bs in batch_sizes:
    start = time.time()
    predict_cell_types(csdata[:100], model, batch_size=bs)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Batch size {bs}: {elapsed:.2f}s")

# Use optimal batch size
optimal_bs = batch_sizes[np.argmin(times)]
print(f"\nOptimal batch size: {optimal_bs}")
```

---

## Integration with Scanpy

### Adding Predictions to AnnData

```python
import scanpy as sc

# Load and predict
adata = sc.read_h5ad('my_data.h5ad')
csdata = C2SData.from_anndata(adata)
model = C2SModel()

predictions = predict_cell_types(csdata, model, n_genes=200)

# Add to AnnData
adata.obs['c2s_prediction'] = predictions

# Visualize
sc.pl.umap(adata, color=['c2s_prediction', 'cell_type'])
```

### Using Embeddings in Scanpy

```python
# Generate embeddings
embeddings = embed_cells(csdata, model, n_genes=200)

# Add to AnnData
adata.obsm['X_c2s'] = embeddings

# Use for neighbors and UMAP
sc.pp.neighbors(adata, use_rep='X_c2s')
sc.tl.umap(adata)
sc.tl.leiden(adata)

# Plot
sc.pl.umap(adata, color=['leiden', 'cell_type'])
```

### Comparative Analysis

```python
# Compare C2S embeddings with PCA
sc.tl.pca(adata)
sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.umap(adata)

adata.obsm['X_umap_pca'] = adata.obsm['X_umap'].copy()

# Now with C2S
sc.pp.neighbors(adata, use_rep='X_c2s')
sc.tl.umap(adata)

# Plot both
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sc.pl.umap(adata, color='cell_type', ax=ax1, show=False, title='PCA-based UMAP')
adata.obsm['X_umap_temp'] = adata.obsm['X_umap'].copy()
adata.obsm['X_umap'] = adata.obsm['X_umap_pca']
sc.pl.umap(adata, color='cell_type', ax=ax2, show=False, title='C2S-based UMAP')
plt.show()
```

---

## Common Workflows

### Workflow 1: Cell Type Annotation

```python
import scanpy as sc
from c2s_mini import C2SData, C2SModel, predict_cell_types

# 1. Load unannotated data
adata = sc.read_h5ad('unannotated_data.h5ad')

# 2. Basic preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 3. Convert and predict
csdata = C2SData.from_anndata(adata)
model = C2SModel()
predictions = predict_cell_types(csdata, model, n_genes=200)

# 4. Add to AnnData
adata.obs['cell_type'] = predictions

# 5. Visualize
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color='cell_type')

# 6. Save
adata.write_h5ad('annotated_data.h5ad')
```

### Workflow 2: Synthetic Data Generation

```python
from c2s_mini import C2SModel, generate_cells, C2SData
from c2s_mini.utils import reconstruct_expression_from_cell_sentence
import anndata
import pandas as pd

# 1. Define composition
cell_types = ['T cell'] * 500 + ['B cell'] * 300 + ['NK cell'] * 200

# 2. Generate sentences
model = C2SModel()
generated = generate_cells(cell_types, model, n_genes=300)

# 3. Reconstruct expression
vocab = [...] # Your vocabulary
expressions = []

for sentence in generated:
    expr = reconstruct_expression_from_cell_sentence(
        sentence, ' ', vocab, slope=-0.5, intercept=3.0
    )
    expressions.append(expr)

# 4. Create AnnData
synthetic_adata = anndata.AnnData(
    X=np.array(expressions),
    obs=pd.DataFrame({'cell_type': cell_types}),
    var=pd.DataFrame(index=vocab)
)

# 5. Use for testing/benchmarking
print(f"Created {synthetic_adata.n_obs} synthetic cells")
```

### Workflow 3: Cross-Dataset Comparison

```python
import scanpy as sc
from c2s_mini import C2SData, C2SModel, embed_cells

# Load datasets
adata1 = sc.read_h5ad('dataset1.h5ad')
adata2 = sc.read_h5ad('dataset2.h5ad')

# Convert
csdata1 = C2SData.from_anndata(adata1, include_obs_columns=['cell_type'])
csdata2 = C2SData.from_anndata(adata2, include_obs_columns=['cell_type'])

# Embed
model = C2SModel()
emb1 = embed_cells(csdata1, model, n_genes=200)
emb2 = embed_cells(csdata2, model, n_genes=200)

# Combine
combined_emb = np.vstack([emb1, emb2])
labels = ['Dataset1'] * len(emb1) + ['Dataset2'] * len(emb2)

# Visualize
import umap

reducer = umap.UMAP()
coords = reducer.fit_transform(combined_emb)

plt.scatter(coords[:, 0], coords[:, 1], c=[labels], alpha=0.5)
plt.legend()
plt.title('Cross-dataset comparison')
plt.show()
```

---

## Next Steps

- Check [API.md](API.md) for complete API reference
- See [examples/](../examples/) for Jupyter notebooks
- Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Join discussions at [GitHub](https://github.com/salzcamino/c2s-mini/discussions)
