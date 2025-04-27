import pytest
import torch

from TransformerMS.Model.model_architecture import GPTLanguageModel
from transformers import AutoTokenizer

@pytest.fixture
def sample_model():
    return GPTLanguageModel(
        vocab_size=1000,
        block_size=32,
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.1
    )

def test_model_initialization(sample_model):
    assert sample_model.token_embedding_table.weight.shape[0] == 1000
    assert len(sample_model.blocks) == 2

def test_model_forward(sample_model):
    import torch
    dummy_input = torch.randint(0, 1000, (1, 10))
    logits, loss = sample_model(dummy_input)
    assert logits.shape == (1, 10, 1000)
    assert loss is None

def test_generation(sample_model):
    generated = sample_model.generate(torch.tensor([[0]]), max_new_tokens=5)
    assert generated.shape == (1, 6)


