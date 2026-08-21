# UPLOTS

UPLOTS is a prompt-conditioned, channel-independent diffusion model for generating time-series trajectories from several predefined dataset/scenario pools with one shared model. A short prompt identifier is embedded by a pretrained language backbone, concatenated with the noisy time-series embedding, and processed at every reverse-diffusion step. The projected hidden states therefore condition the Diffusion-TS denoiser that estimates the clean sequence.

The implementation supports GPT-2 and LLaMA as interchangeable language backbones. GPT-2 remains the default so existing commands and checkpoints continue to work.

The runnable project is under `UPLOTS/`; the `baselines/` directory contains the reproduced baseline implementations.

## Setup

Create a Python environment, install a CUDA-compatible PyTorch build, and install the remaining dependencies:

```bash
cd UPLOTS
pip install -r requirements.txt
```

The default LLaMA model, `meta-llama/Llama-3.2-1B`, is gated. Accept its license and authenticate through the Hugging Face CLI before training. Do not put an access token in a command or log file.

Prepare the scenario CSV files under `Data/datasets/`. The repository includes notebooks for the original ETTh, Energy, PEMS04, and PEMS08 peak-pattern pools. Configuration names passed to `--config_file` refer to YAML files under `Config/` and should be given without the `.yaml` suffix.

## Backbone options

| Option | Default | Meaning |
|---|---|---|
| `--backbone` | `gpt2` | Select `gpt2` or `llama`. |
| `--backbone_name` | family default | Hugging Face model ID or a local model directory. |
| `--backbone_layers` | GPT-2: 2; LLaMA: 8 | Retain the first N transformer layers. |
| `--backbone_init` | `pretrained` | Use `pretrained` or `random` backbone weights. |
| `--prompt_mode` | `text` | Use tokenized `text`, a `learned_id`, or `none`. |
| `--train_backbone` | off | Fine-tune all backbone parameters. |

Under the default mostly-frozen policy, GPT-2 LayerNorm and learned positional-embedding parameters remain trainable. For LLaMA, RMSNorm parameters remain trainable; its rotary positional encoding is parameter-free. Diffusion-TS, the time-series projections, and prompt/output projections are trained in both cases.

## Training

Run the following commands from the `UPLOTS/` directory. This example trains one model on the four original peak-pattern pools:

```bash
python main.py \
  --name uplots_gpt2_peak4 \
  --config_file morning_peak_etth evening_peak_etth morning_peak_energy evening_peak_energy \
  --backbone gpt2 \
  --train --epoch 1000 --batch 32 --checkpoint_every 250 --gpu 0
```

Select the LLaMA option by changing the run name and backbone arguments:

```bash
huggingface-cli login

python main.py \
  --name uplots_llama_peak4 \
  --config_file morning_peak_etth evening_peak_etth morning_peak_energy evening_peak_energy \
  --backbone llama \
  --backbone_name meta-llama/Llama-3.2-1B \
  --backbone_layers 8 \
  --train --epoch 1000 --batch 32 --checkpoint_every 250 --gpu 0
```

The 14-pool ETTh/Energy setting uses:

```text
morning_peak_etth evening_peak_etth morning_peak_energy evening_peak_energy
workday_etth weekend_etth workday_energy weekend_energy
high_load_etth low_load_etth high_load_energy low_load_energy
volatile_etth volatile_energy
```

Training pools every variable trajectory as a separate sequence before it enters the shared model. This permits datasets with different channel counts to share one generator while keeping generated arrays in `[samples, time, 1]` format.

## Sampling

Use the same run name and all backbone/prompt options that were used for training. Sampling one requested pool requires exactly one configuration:

```bash
python main.py \
  --name uplots_llama_peak4 \
  --config_file morning_peak_etth \
  --backbone llama \
  --backbone_name meta-llama/Llama-3.2-1B \
  --backbone_layers 8 \
  --milestone 1000 --num_samples 4096 --sample_batch_size 256 --gpu 0
```

Legacy GPT-2 checkpoints retain their original `gpt2_tr.*` state-dict keys and load with the default backbone arguments. A checkpoint trained with LLaMA must be sampled with the matching model ID/path and layer count.

## CALM and RLDS controls

Curriculum-Aware Loss Modulation (CALM) downweights higher-loss pools during its active curriculum interval. Rolling-Loss Dynamic Sampling (RLDS) samples pool indices in proportion to their rolling losses and restarts exhausted dataloaders as needed.

Relevant options include `--disable_calm`, `--disable_rlds`, `--calm_warmup_epochs`, `--calm_active_epochs`, `--calm_min_weight`, `--loss_window`, and `--rlds_temperature`.

## Verification

Run the offline unit tests:

```bash
python -m unittest tests.test_language_backbone tests.test_training_strategy -v
```

Run a real end-to-end backbone/denoiser forward pass. The tiny LLaMA checkpoint below is for structural testing only, not for reported experiments:

```bash
python tests/smoke_backbone.py --backbone gpt2

python tests/smoke_backbone.py \
  --backbone llama \
  --backbone-name hf-internal-testing/tiny-random-LlamaForCausalLM \
  --backbone-layers 2
```

## Evaluation

`evaluate.py` computes Context-FID, discriminative, and predictive scores after deterministic channel pooling and equal-count balancing. `Experiments/constraint_satisfaction.py` computes the Scenario Preference Score (SPS), including bootstrap intervals over generated/reference samples. These intervals describe sampling variability and are not independent training-seed intervals.

## Acknowledgement

UPLOTS builds its diffusion denoiser and several evaluation components on [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS). Please also follow the attribution and licensing requirements of the upstream projects and pretrained backbone providers.
