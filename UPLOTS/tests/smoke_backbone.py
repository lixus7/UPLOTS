#!/usr/bin/env python
"""Run one CPU forward pass through the selected UPLOTS backbone and denoiser."""

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['gpt2', 'llama'], default='gpt2')
    parser.add_argument('--backbone-name', default=None)
    parser.add_argument('--backbone-layers', type=int, default=None)
    parser.add_argument('--prompt', default='ETTMP')
    return parser.parse_args()


def main():
    args = parse_args()
    model = Diffusion_TS(
        seq_length=24,
        feature_size=1,
        n_layer_enc=1,
        n_layer_dec=1,
        d_model=8,
        timesteps=4,
        sampling_timesteps=4,
        n_heads=2,
        backbone=args.backbone,
        backbone_name=args.backbone_name,
        backbone_layers=args.backbone_layers,
        prompt_mode='text',
    ).cpu().eval()

    sample = torch.randn(2, 24, 1)
    timesteps = torch.tensor([1, 2])
    with torch.no_grad():
        trend, season = model.model(args.prompt, sample, timesteps)

    print(json.dumps({
        'backbone': model.model.backbone,
        'model_name': model.model.backbone_name,
        'layers': model.model.backbone_layers,
        'trend_shape': list(trend.shape),
        'season_shape': list(season.shape),
    }, indent=2))


if __name__ == '__main__':
    main()
