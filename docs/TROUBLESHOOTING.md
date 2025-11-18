# Troubleshooting Guide

Common issues and solutions for c2s-mini.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Model Loading Issues](#model-loading-issues)
- [Memory Issues](#memory-issues)
- [Performance Issues](#performance-issues)
- [Data Format Issues](#data-format-issues)
- [Prediction Quality Issues](#prediction-quality-issues)
- [Common Errors](#common-errors)

---

## Installation Issues

### Problem: `pip install -e .` fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```

**Solutions:**

1. **Install PyTorch first:**
   ```bash
   # CPU only
   pip install torch --index-url https://download.pytorch.org/whl/cpu

   # CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # Then install c2s-mini
   pip install -e .
   ```

2. **Use conda:**
   ```bash
   conda install pytorch -c pytorch
   pip install -e .
   ```

### Problem: `ModuleNotFoundError: No module named 'c2s_mini'`

**Symptoms:**
```python
>>> import c2s_mini
ModuleNotFoundError: No module named 'c2s_mini'
```

**Solutions:**

1. **Verify installation:**
   ```bash
   pip list | grep c2s-mini
   ```

2. **Reinstall in editable mode:**
   ```bash
   cd /path/to/c2s-mini
   pip install -e .
   ```

3. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   ```

### Problem: Missing scikit-learn dependency

**Symptoms:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**
```bash
pip install scikit-learn>=1.0.0
```

Or reinstall c2s-mini to get all dependencies:
```bash
pip install -e . --upgrade
```

---

## Model Loading Issues

### Problem: Model download fails

**Symptoms:**
```
OSError: Can't load model 'vandijklab/pythia-160m-c2s'
```

**Solutions:**

1. **Check internet connection**

2. **Try manual download:**
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM

   # Download to cache
   tokenizer = AutoTokenizer.from_pretrained("vandijklab/pythia-160m-c2s")
   model = AutoModelForCausalLM.from_pretrained("vandijklab/pythia-160m-c2s")

   # Then use C2SModel
   from c2s_mini import C2SModel
   model = C2SModel()
   ```

3. **Set HuggingFace cache directory:**
   ```bash
   export HF_HOME=/path/to/cache
   export TRANSFORMERS_CACHE=/path/to/cache
   ```

4. **Use offline mode (if model already downloaded):**
   ```python
   import os
   os.environ['TRANSFORMERS_OFFLINE'] = '1'
   ```

### Problem: CUDA out of memory during model loading

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Use CPU instead:**
   ```python
   model = C2SModel(device='cpu')
   ```

2. **Clear CUDA cache before loading:**
   ```python
   import torch
   torch.cuda.empty_cache()

   model = C2SModel(device='cuda')
   ```

3. **Close other GPU processes:**
   ```bash
   nvidia-smi  # Check GPU usage
   ```

### Problem: Slow model loading

**Symptoms:**
- Model takes >5 minutes to load

**Solutions:**

1. **First load is always slow** (downloading ~300MB)
   - Subsequent loads use cached model

2. **Check disk space:**
   ```bash
   df -h ~/.cache/huggingface
   ```

3. **Use SSD for cache:**
   ```bash
   export HF_HOME=/path/to/ssd/cache
   ```

---

## Memory Issues

### Problem: Out of memory during inference

**Symptoms:**
```
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   predictions = predict_cell_types(
       csdata,
       model,
       batch_size=2  # Reduce from default 8
   )
   ```

2. **Reduce number of genes:**
   ```python
   predictions = predict_cell_types(
       csdata,
       model,
       n_genes=50  # Reduce from default 200
   )
   ```

3. **Process in chunks:**
   ```python
   chunk_size = 100
   all_predictions = []

   for i in range(0, len(csdata), chunk_size):
       chunk = C2SData(
           vocab=csdata.vocab,
           sentences=csdata.sentences[i:i+chunk_size]
       )
       preds = predict_cell_types(chunk, model, batch_size=2)
       all_predictions.extend(preds)

       # Clear cache
       import torch
       if model.device == 'cuda':
           torch.cuda.empty_cache()
   ```

4. **Use CPU:**
   ```python
   model = C2SModel(device='cpu')
   ```

### Problem: Memory leak during long-running processes

**Symptoms:**
- Memory usage grows continuously
- System becomes unresponsive

**Solutions:**

1. **Clear cache periodically:**
   ```python
   import torch
   import gc

   for i in range(0, len(csdata), chunk_size):
       # Process chunk...

       # Clear memory
       if model.device == 'cuda':
           torch.cuda.empty_cache()
       gc.collect()
   ```

2. **Reload model periodically:**
   ```python
   for batch_num, batch in enumerate(large_dataset):
       if batch_num % 100 == 0:
           # Reload model
           del model
           torch.cuda.empty_cache()
           model = C2SModel(device='cuda')
   ```

---

## Performance Issues

### Problem: Inference is very slow

**Symptoms:**
- Takes >1 second per cell
- Slower than expected

**Solutions:**

1. **Use GPU if available:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")

   model = C2SModel(device='cuda')  # Use GPU
   ```

2. **Increase batch size:**
   ```python
   predictions = predict_cell_types(
       csdata,
       model,
       batch_size=32 if model.device == 'cuda' else 4
   )
   ```

3. **Reduce max_tokens:**
   ```python
   predictions = predict_cell_types(
       csdata,
       model,
       max_tokens=30  # Instead of default 1024
   )
   ```

4. **Use fewer genes:**
   ```python
   predictions = predict_cell_types(
       csdata,
       model,
       n_genes=100  # Instead of 200
   )
   ```

### Problem: Progress bars don't show

**Symptoms:**
- No tqdm progress bars visible

**Solutions:**

1. **Install tqdm:**
   ```bash
   pip install tqdm
   ```

2. **Disable if problematic:**
   ```python
   # Monkey-patch tqdm
   from tqdm import tqdm as original_tqdm
   def tqdm(iterable, *args, **kwargs):
       return iterable
   ```

---

## Data Format Issues

### Problem: `ValueError: Metadata has X rows but there are Y sentences`

**Symptoms:**
```
ValueError: Metadata has 100 rows but there are 200 sentences. They must match.
```

**Solution:**

Ensure metadata matches number of cells:

```python
# Wrong
metadata = adata.obs.head(100)  # Only 100 rows
sentences = ['...'] * 200      # 200 sentences

# Right
metadata = adata.obs  # Same as number of cells
sentences = ['...'] * len(adata.obs)
```

### Problem: `ValueError: Requested columns not found in adata.obs`

**Symptoms:**
```
ValueError: Requested columns ['cell_type'] not found in adata.obs
```

**Solution:**

Check available columns:

```python
# Check what's available
print(adata.obs.columns)

# Use correct column names
csdata = C2SData.from_anndata(
    adata,
    include_obs_columns=['leiden']  # Use actual column name
)
```

### Problem: Gene names are Ensembl IDs

**Symptoms:**
```
WARN: adata.var_names seems to contain Ensembl IDs rather than gene/feature names
```

**Solution:**

Convert to gene symbols:

```python
# If you have a mapping
gene_mapping = {
    'ENSG00000000003': 'TSPAN6',
    'ENSG00000000005': 'TNMD',
    # ...
}

adata.var_names = [gene_mapping.get(name, name) for name in adata.var_names]

# Or use biomaRt (R) or mygene (Python)
import mygene
mg = mygene.MyGeneInfo()
results = mg.querymany(adata.var_names, scopes='ensembl.gene')
```

### Problem: Sparse matrix conversion warnings

**Symptoms:**
```
UserWarning: Converting sparse matrix to dense
```

**Solution:**

c2s-mini handles sparse matrices efficiently. If you see this warning, check your code:

```python
# Good - keeps sparse
csdata = C2SData.from_anndata(adata)

# Bad - converts to dense
adata.X = adata.X.toarray()  # Don't do this
```

---

## Prediction Quality Issues

### Problem: Predictions are all the same

**Symptoms:**
- All cells predicted as same type
- No diversity in predictions

**Solutions:**

1. **Use sampling:**
   ```python
   predictions = predict_cell_types(
       csdata,
       model,
       do_sample=True,
       temperature=1.0
   )
   ```

2. **Increase number of genes:**
   ```python
   predictions = predict_cell_types(
       csdata,
       model,
       n_genes=300  # More context
   )
   ```

3. **Check data quality:**
   ```python
   # Check if cells are actually different
   sentences = csdata.get_sentences()
   print(f"Unique sentences: {len(set(sentences))}")

   # Check top genes
   for i in range(5):
       top_genes = ' '.join(sentences[i].split()[:10])
       print(f"Cell {i}: {top_genes}")
   ```

### Problem: Predictions don't match expectations

**Symptoms:**
- Predicted cell types seem wrong
- Low agreement with ground truth

**Solutions:**

1. **Check organism parameter:**
   ```python
   # Make sure organism matches your data
   predictions = predict_cell_types(
       csdata,
       model,
       organism='Homo sapiens'  # or 'Mus musculus'
   )
   ```

2. **Tune temperature:**
   ```python
   # More conservative
   predictions = predict_cell_types(
       csdata,
       model,
       temperature=0.5
   )
   ```

3. **Use more genes:**
   ```python
   predictions = predict_cell_types(
       csdata,
       model,
       n_genes=500
   )
   ```

4. **Check gene naming:**
   ```python
   # Gene names should be uppercase symbols
   vocab = csdata.get_vocab()
   print(f"First 10 genes: {list(vocab.keys())[:10]}")

   # Should see: ['CD3D', 'CD3E', ...]
   # Not: ['ENSG00000...'] or ['cd3d', 'cd3e', ...]
   ```

### Problem: Generated cells are nonsensical

**Symptoms:**
- Generated sentences contain non-gene words
- Genes repeated multiple times

**Solution:**

Use post-processing:

```python
from c2s_mini.utils import post_process_generated_cell_sentences

generated = generate_cells(['T cell'], model, n_genes=200)[0]
vocab_list = list(csdata.get_vocab().keys())

cleaned, n_replaced = post_process_generated_cell_sentences(
    generated,
    vocab_list
)

print(f"Cleaned sentence: {' '.join(cleaned)}")
print(f"Replaced {n_replaced} invalid words")
```

---

## Common Errors

### `RuntimeError: CUDA requested but not available`

**Cause:** Trying to use CUDA without GPU

**Solution:**
```python
# Use auto-detect
model = C2SModel(device='auto')

# Or force CPU
model = C2SModel(device='cpu')
```

### `TypeError: 'NoneType' object is not iterable`

**Cause:** Missing or None data

**Solution:**

Check your data:

```python
# Check csdata
print(f"Sentences: {len(csdata.get_sentences())}")
print(f"Vocab: {len(csdata.get_vocab())}")

# Check for None
if csdata.sentences is None:
    print("ERROR: No sentences!")
```

### `IndexError: list index out of range`

**Cause:** Empty data or mismatched sizes

**Solution:**

```python
# Check data size
print(f"Dataset size: {len(csdata)}")
print(f"Number of sentences: {len(csdata.get_sentences())}")

# Don't try to access empty data
if len(csdata) > 0:
    first_sentence = csdata.get_sentences()[0]
```

### `KeyError: 'cell_type'`

**Cause:** Accessing metadata that doesn't exist

**Solution:**

```python
# Check if metadata exists
if csdata.metadata is not None:
    if 'cell_type' in csdata.metadata.columns:
        cell_types = csdata.metadata['cell_type']
    else:
        print(f"Available columns: {csdata.metadata.columns}")
else:
    print("No metadata available")
```

---

## Debugging Tips

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('c2s_mini')
logger.setLevel(logging.DEBUG)
```

### Check Tensor Devices

```python
# Make sure tensors are on correct device
def check_device(tensor):
    print(f"Tensor device: {tensor.device}")
    print(f"Model device: {model.device}")
    assert str(tensor.device) == model.device, "Device mismatch!"
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()

# Your code here
predictions = predict_cell_types(csdata, model)

# Check memory usage
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

### GPU Memory Tracking

```python
import torch

if torch.cuda.is_available():
    # Before operation
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Your operation
    predictions = predict_cell_types(csdata, model)

    # After operation
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Minimal Reproducible Example

When reporting issues, create a minimal example:

```python
import scanpy as sc
from c2s_mini import C2SData, C2SModel, predict_cell_types

# Use small public dataset
adata = sc.datasets.pbmc3k()
adata = adata[:10, :50]  # Tiny subset

# Reproduce issue
csdata = C2SData.from_anndata(adata)
model = C2SModel(device='cpu')

try:
    predictions = predict_cell_types(csdata, model, batch_size=2)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

---

## Getting Help

If you're still stuck:

1. **Check existing issues:** [GitHub Issues](https://github.com/salzcamino/c2s-mini/issues)
2. **Search discussions:** [GitHub Discussions](https://github.com/salzcamino/c2s-mini/discussions)
3. **Open new issue:** Provide:
   - Python version (`python --version`)
   - Package version (`pip show c2s-mini`)
   - Minimal reproducible example
   - Full error traceback
   - System info (OS, GPU, memory)

**Issue Template:**

```markdown
## Description
Brief description of the problem

## Environment
- Python version:
- c2s-mini version:
- PyTorch version:
- OS:
- GPU (if applicable):

## Minimal Reproducible Example
```python
# Your code here
```

## Error Traceback
```
# Full error message
```

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened
```

---

## FAQ

**Q: How long should model loading take?**

A: First time: 2-10 minutes (downloads ~300MB). Subsequent loads: 10-30 seconds.

**Q: Can I use my own models?**

A: No, c2s-mini only supports `vandijklab/pythia-160m-c2s`. For custom models, use the full Cell2Sentence implementation.

**Q: Why are predictions inconsistent between runs?**

A: Due to sampling. Use `random_state` and `do_sample=False` for deterministic results:

```python
csdata = C2SData.from_anndata(adata, random_state=42)
predictions = predict_cell_types(
    csdata, model,
    do_sample=False  # Deterministic
)
```

**Q: Can I train/fine-tune the model?**

A: No, c2s-mini is inference-only. Use the full Cell2Sentence package for training.

**Q: How do I cite c2s-mini?**

A: Cite the original Cell2Sentence paper (see README.md).
