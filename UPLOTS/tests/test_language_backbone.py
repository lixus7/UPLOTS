import unittest
from types import SimpleNamespace

import torch
from torch import nn

from Models.interpretable_diffusion.language_backbone import (
    configure_backbone_trainability,
    hidden_states_from_output,
    resolve_backbone_spec,
)


class TinyGPT2Like(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(8, 4)
        self.wpe = nn.Embedding(8, 4)
        self.ln_f = nn.LayerNorm(4)
        self.attn = nn.Linear(4, 4)


class TinyLlamaLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


class LanguageBackboneTest(unittest.TestCase):
    def test_default_specs(self):
        self.assertEqual(resolve_backbone_spec('GPT2'), ('gpt2', 'gpt2', 2))
        self.assertEqual(
            resolve_backbone_spec('llama'),
            ('llama', 'meta-llama/Llama-3.2-1B', 8),
        )

    def test_invalid_spec(self):
        with self.assertRaises(ValueError):
            resolve_backbone_spec('bert')
        with self.assertRaises(ValueError):
            resolve_backbone_spec('gpt2', num_layers=0)

    def test_gpt2_frozen_policy(self):
        model = TinyGPT2Like()
        configure_backbone_trainability(model, 'gpt2')
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertEqual(
            trainable,
            {'wpe.weight', 'ln_f.weight', 'ln_f.bias'},
        )

    def test_llama_frozen_policy(self):
        model = TinyLlamaLike()
        configure_backbone_trainability(model, 'llama')
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertEqual(
            trainable,
            {
                'input_layernorm.weight',
                'input_layernorm.bias',
                'norm.weight',
                'norm.bias',
            },
        )

    def test_trainable_policy(self):
        model = TinyLlamaLike()
        configure_backbone_trainability(model, 'llama', train_backbone=True)
        self.assertTrue(all(value.requires_grad for value in model.parameters()))

    def test_output_normalization(self):
        expected = torch.randn(2, 3, 4)
        self.assertIs(hidden_states_from_output(expected), expected)
        self.assertIs(
            hidden_states_from_output(SimpleNamespace(last_hidden_state=expected)),
            expected,
        )
        self.assertIs(hidden_states_from_output((expected,)), expected)
        with self.assertRaises(TypeError):
            hidden_states_from_output(SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
