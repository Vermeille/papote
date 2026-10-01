import torch
import torch.nn.functional as F

from papote.model import Transformer


def test_transformer_linear_cross_entropy_matches_materialized_logits():
    torch.manual_seed(0)
    model = Transformer(
        num_tokens=17,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        head_size=4,
        context_size=6,
    )
    input_ids = torch.randint(0, 17, (2, 6))
    output_ids = torch.randint(0, 17, (2, 6))

    logits = model(input_ids)
    expected = F.cross_entropy(
        logits.transpose(1, 2), output_ids, reduction="none"
    )
    actual = model(input_ids, output_ids=output_ids)

    assert actual.shape == output_ids.shape
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_transformer_linear_cross_entropy_backpropagates():
    torch.manual_seed(0)
    model = Transformer(
        num_tokens=17,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        head_size=4,
        context_size=6,
    )
    input_ids = torch.randint(0, 17, (2, 6))
    output_ids = torch.randint(0, 17, (2, 6))

    model(input_ids, output_ids=output_ids).mean().backward()

    assert model.token_embedding.unembed.weight.grad is not None
    assert model.transformer_blocks[0].sa.qkv.weight.grad is not None
