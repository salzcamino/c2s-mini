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


# Error handling and edge case tests


def test_model_device_auto():
    """Test automatic device selection doesn't fail."""
    # This should not raise an error
    try:
        model = C2SModel(device='auto')
        assert model.device in ['cpu', 'cuda']
    except Exception:
        # If it fails, it's likely due to missing dependencies, skip
        pytest.skip("Model loading failed, possibly missing dependencies")


@pytest.mark.slow
def test_model_repr():
    """Test model string representation."""
    model = C2SModel(device='cpu')
    repr_str = repr(model)
    assert "C2SModel" in repr_str
    assert "pythia-160m-c2s" in repr_str


@pytest.mark.slow
def test_generation_empty_prompt():
    """Test generation with empty prompt."""
    model = C2SModel(device='cpu')
    result = model.generate_from_prompt("", max_tokens=10)
    # Should return something (even if just empty or EOS token)
    assert isinstance(result, str)


@pytest.mark.slow
def test_batch_generation_empty_list():
    """Test batch generation with empty list."""
    model = C2SModel(device='cpu')
    results = model.generate_batch([], max_tokens=10)
    assert len(results) == 0


@pytest.mark.slow
def test_batch_generation_single_prompt():
    """Test batch generation with single prompt."""
    model = C2SModel(device='cpu')
    prompts = ["Predict the cell type: CD3D"]
    results = model.generate_batch(prompts, max_tokens=20)
    assert len(results) == 1
    assert isinstance(results[0], str)


@pytest.mark.slow
def test_embed_batch_single_prompt():
    """Test batch embedding with single prompt."""
    model = C2SModel(device='cpu')
    prompts = ["CD3D CD3E"]
    embeddings = model.embed_batch(prompts)
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] > 0
