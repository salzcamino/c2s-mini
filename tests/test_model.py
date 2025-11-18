import pytest
from c2s_mini.model import C2SModel


@pytest.mark.slow
def test_model_loading():
    """Test model loads successfully."""
    model = C2SModel(device='cpu')
    assert model.model is not None
    assert model.tokenizer is not None


@pytest.mark.slow
def test_generation():
    """Test text generation."""
    model = C2SModel(device='cpu')
    prompt = "Predict the cell type: CD3D CD3E CD8A"

    result = model.generate_from_prompt(prompt, max_tokens=20)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.slow
def test_batch_generation():
    """Test batch generation."""
    model = C2SModel(device='cpu')
    prompts = [
        "Predict the cell type: CD3D CD3E",
        "Predict the cell type: CD19 MS4A1"
    ]

    results = model.generate_batch(prompts, max_tokens=20)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


@pytest.mark.slow
def test_embedding():
    """Test cell embedding."""
    model = C2SModel(device='cpu')
    prompt = "CD3D CD3E CD8A"

    embedding = model.embed_cell(prompt)
    assert embedding.shape[0] > 0  # Has some dimension
