"""Language-backbone construction for prompt-conditioned Diffusion-TS.

The public interface deliberately keeps GPT-2 as the default so existing
commands and checkpoints continue to work.  LLaMA uses Hugging Face's base
``AutoModel`` (not its autoregressive LM head), because UPLOTS consumes hidden
states and leaves sequence denoising to Diffusion-TS.
"""

import torch

from transformers import AutoConfig, AutoModel, AutoTokenizer, GPT2Config, GPT2Tokenizer

from Models.interpretable_diffusion.unitimegpt2 import UniTimeGPT2


DEFAULT_MODEL_NAMES = {
    'gpt2': 'gpt2',
    'llama': 'meta-llama/Llama-3.2-1B',
}
DEFAULT_LAYER_COUNTS = {
    'gpt2': 2,
    'llama': 8,
}


def resolve_backbone_spec(backbone, model_name=None, num_layers=None):
    """Return a validated ``(backbone, model_name, num_layers)`` tuple."""
    backbone = str(backbone).lower()
    if backbone not in DEFAULT_MODEL_NAMES:
        raise ValueError(
            f'Unknown backbone {backbone!r}; choose from {sorted(DEFAULT_MODEL_NAMES)}.'
        )
    model_name = model_name or DEFAULT_MODEL_NAMES[backbone]
    num_layers = DEFAULT_LAYER_COUNTS[backbone] if num_layers is None else int(num_layers)
    if num_layers <= 0:
        raise ValueError('backbone_layers must be positive.')
    return backbone, model_name, num_layers


def _validate_layer_count(config, num_layers, model_name):
    available = getattr(config, 'num_hidden_layers', None)
    if available is None:
        available = getattr(config, 'n_layer', None)
    if available is None:
        raise ValueError(f'Cannot determine the layer count for {model_name!r}.')
    if num_layers > int(available):
        raise ValueError(
            f'backbone_layers={num_layers} exceeds the {available} layers in {model_name!r}.'
        )


def configure_backbone_trainability(model, backbone, train_backbone=False):
    """Apply the paper's mostly-frozen policy to either supported backbone."""
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if train_backbone:
            parameter.requires_grad = True
        elif backbone == 'gpt2':
            parameter.requires_grad = 'ln' in lowered or 'wpe' in lowered
        else:
            # LLaMA uses RMSNorm and rotary (parameter-free) positional encoding.
            parameter.requires_grad = 'norm' in lowered


def load_language_backbone(
    backbone='gpt2',
    model_name=None,
    num_layers=None,
    backbone_init='pretrained',
    prompt_mode='text',
    train_backbone=False,
):
    """Build a GPT-2 or LLaMA base model and its optional text tokenizer.

    Returns ``(model, tokenizer, hidden_size, resolved_model_name,
    resolved_num_layers)``.  The LLaMA default is gated on Hugging Face; users
    should authenticate outside the command line so credentials never enter
    saved argument logs.
    """
    backbone, model_name, num_layers = resolve_backbone_spec(
        backbone, model_name=model_name, num_layers=num_layers
    )
    if backbone_init not in {'pretrained', 'random'}:
        raise ValueError("backbone_init must be either 'pretrained' or 'random'.")
    if prompt_mode not in {'text', 'learned_id', 'none'}:
        raise ValueError("prompt_mode must be 'text', 'learned_id', or 'none'.")

    try:
        if backbone == 'gpt2':
            config = GPT2Config.from_pretrained(model_name)
            _validate_layer_count(config, num_layers, model_name)
            tokenizer = (
                GPT2Tokenizer.from_pretrained(model_name) if prompt_mode == 'text' else None
            )
            if backbone_init == 'pretrained':
                model = UniTimeGPT2.from_pretrained(model_name)
                model.transformer.h = model.transformer.h[:num_layers]
                model.config.n_layer = num_layers
            else:
                config.n_layer = num_layers
                model = UniTimeGPT2(config)
            hidden_size = int(model.config.n_embd)
        else:
            config = AutoConfig.from_pretrained(model_name)
            if getattr(config, 'model_type', None) != 'llama':
                raise ValueError(
                    f'--backbone llama requires a LLaMA checkpoint, but {model_name!r} '
                    f'has model_type={getattr(config, "model_type", None)!r}.'
                )
            _validate_layer_count(config, num_layers, model_name)
            config.num_hidden_layers = num_layers
            tokenizer = (
                AutoTokenizer.from_pretrained(model_name, use_fast=True)
                if prompt_mode == 'text'
                else None
            )
            model = (
                AutoModel.from_pretrained(model_name, config=config)
                if backbone_init == 'pretrained'
                else AutoModel.from_config(config)
            )
            hidden_size = int(config.hidden_size)
    except OSError as exc:
        hint = ''
        if backbone == 'llama':
            hint = (
                ' Accept the Meta LLaMA license and authenticate with '
                '`huggingface-cli login`, or pass a local model path with '
                '`--backbone_name`.'
            )
        raise RuntimeError(f'Could not load language backbone {model_name!r}.{hint}') from exc

    model.config.use_cache = False
    configure_backbone_trainability(model, backbone, train_backbone=train_backbone)
    return model, tokenizer, hidden_size, model_name, num_layers


def hidden_states_from_output(output):
    """Normalize Hugging Face and legacy UniTimeGPT2 outputs to one tensor."""
    if torch.is_tensor(output):
        return output
    hidden_states = getattr(output, 'last_hidden_state', None)
    if hidden_states is not None:
        return hidden_states
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f'Unsupported backbone output type: {type(output)!r}')
