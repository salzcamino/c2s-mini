"""
Model wrapper for Cell2Sentence inference.

This module provides the C2SModel class for loading and using the
pythia-160m-c2s model from HuggingFace for cell type prediction,
cell generation, and cell embedding tasks.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Union
from tqdm import tqdm


MODEL_NAME = "vandijklab/pythia-160m-c2s"


class C2SModel:
    """
    Wrapper for Cell2Sentence model (inference only).

    This class provides a simple interface for loading the pythia-160m-c2s
    model and performing inference tasks including text generation and
    cell embedding.

    Attributes:
        device (str): Device to run the model on ('cuda' or 'cpu')
        tokenizer: HuggingFace tokenizer instance
        model: HuggingFace model instance

    Example:
        >>> model = C2SModel(device='auto')
        >>> prompt = "Predict the cell type: CD3D CD3E CD8A"
        >>> result = model.generate_from_prompt(prompt, max_tokens=50)
    """

    def __init__(self, device: str = 'auto'):
        """
        Load pythia-160m-c2s model from HuggingFace.

        Args:
            device: Device to use for inference. Options:
                - 'auto': Automatically select CUDA if available, else CPU
                - 'cuda': Force CUDA (will error if not available)
                - 'cpu': Force CPU

        Raises:
            RuntimeError: If CUDA is requested but not available
        """
        # Determine device
        if device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        else:
            self.device = device

        print(f"Loading {MODEL_NAME} on {self.device}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            padding_side='left',
            trust_remote_code=True
        )

        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"Model loaded successfully on {self.device}")

    def generate_from_prompt(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from a single prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate (default: 1024)
            temperature: Sampling temperature (default: 1.0)
            top_p: Nucleus sampling parameter (default: 1.0)
            do_sample: Whether to use sampling (default: True)
            **kwargs: Additional generation parameters passed to model.generate()

        Returns:
            Generated text (excluding the input prompt)

        Example:
            >>> model = C2SModel()
            >>> prompt = "Predict the cell type: CD3D CD3E"
            >>> result = model.generate_from_prompt(prompt, max_tokens=50)
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_length = inputs['input_ids'].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode only the generated part (exclude input prompt)
        generated_tokens = outputs[0, input_length:]
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return generated_text

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text from multiple prompts (batched).

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt (default: 1024)
            temperature: Sampling temperature (default: 1.0)
            top_p: Nucleus sampling parameter (default: 1.0)
            do_sample: Whether to use sampling (default: True)
            **kwargs: Additional generation parameters passed to model.generate()

        Returns:
            List of generated texts (excluding input prompts)

        Example:
            >>> model = C2SModel()
            >>> prompts = ["Predict: CD3D CD3E", "Predict: CD19 MS4A1"]
            >>> results = model.generate_batch(prompts, max_tokens=50)
        """
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Track input lengths for each prompt
        input_lengths = (inputs['attention_mask'].sum(dim=1)).tolist()

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode each output, excluding the input prompt for each
        generated_texts = []
        for i, output in enumerate(outputs):
            # Extract only the generated tokens (after the input)
            generated_tokens = output[input_lengths[i]:]
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)

        return generated_texts

    def embed_cell(self, prompt: str) -> np.ndarray:
        """
        Get embedding for a single cell.

        This method computes the embedding by taking the last hidden state
        from the model at the last token position.

        Args:
            prompt: Formatted cell sentence prompt

        Returns:
            Embedding vector as numpy array (shape: [hidden_size])

        Example:
            >>> model = C2SModel()
            >>> prompt = "CD3D CD3E CD8A CD8B"
            >>> embedding = model.embed_cell(prompt)
            >>> print(embedding.shape)
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model output
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        # Extract last hidden state
        # Shape: [batch_size, sequence_length, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]

        # Get the embedding at the last token position
        # Use the last non-padding token
        sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
        embedding = last_hidden_state[0, sequence_lengths[0], :]

        # Convert to numpy
        return embedding.cpu().numpy()

    def embed_batch(self, prompts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple cells (batched).

        This method computes embeddings for multiple cells efficiently
        using batched processing.

        Args:
            prompts: List of formatted cell sentence prompts

        Returns:
            Array of embeddings (shape: [n_cells, hidden_size])

        Example:
            >>> model = C2SModel()
            >>> prompts = ["CD3D CD3E", "CD19 MS4A1"]
            >>> embeddings = model.embed_batch(prompts)
            >>> print(embeddings.shape)
        """
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get sequence lengths (position of last non-padding token)
        sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1

        # Get model output
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        # Extract last hidden state
        # Shape: [batch_size, sequence_length, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]

        # Extract embeddings at last token position for each sequence
        embeddings = []
        for i, seq_len in enumerate(sequence_lengths):
            embedding = last_hidden_state[i, seq_len, :]
            embeddings.append(embedding)

        # Stack and convert to numpy
        embeddings = torch.stack(embeddings)
        return embeddings.cpu().numpy()

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"C2SModel(model={MODEL_NAME}, device={self.device})"
