#!/usr/bin/env python
"""Evaluate whether generated sets prefer their requested scenario distribution.

The current benchmark constructs each condition as a scenario-specific subset and
does not retain a within-sequence timestamp mask.  Consequently, a window-level
"peak occurs at 07:00--10:00" hit rate is not identifiable from the saved arrays.
This script instead reports Scenario Preference Score (SPS): the generated set's
Fréchet distance to its target reference is compared with the distance to the
paired contrast reference.  SPS = d_contrast / (d_target + d_contrast), so values
above 50% mean that the generated set is closer to the requested scenario.

Distances are computed on transparent summary features (level, dispersion,
quantiles, first differences, half-window means, and lag autocorrelations).  The
reported intervals are bootstrap intervals over samples, not independent model
training seeds.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm


SCENARIOS = (
    'morning_peak',
    'evening_peak',
    'workday',
    'weekend',
    'high_load',
    'low_load',
    'volatile',
)
REAL_STEMS = {'workday': 'workdays', 'weekend': 'non_workdays'}
CONTRASTS = {
    'morning_peak': 'evening_peak',
    'evening_peak': 'morning_peak',
    'workday': 'weekend',
    'weekend': 'workday',
    'high_load': 'low_load',
    'low_load': 'high_load',
    'volatile': 'non-volatile pool',
}
FAMILIES = {
    'Temporal peak': ('morning_peak', 'evening_peak'),
    'Calendar': ('workday', 'weekend'),
    'Load level': ('high_load', 'low_load'),
    'Volatility': ('volatile',),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('OUTPUT/mix14_etth_energy'))
    parser.add_argument('--datasets', nargs='+', default=['etth', 'energy'])
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument('--milestone', type=int, default=1000)
    parser.add_argument('--length', type=int, default=24)
    parser.add_argument('--pool-size', type=int, default=10000)
    parser.add_argument('--bootstrap-size', type=int, default=3000)
    parser.add_argument('--bootstrap-repeats', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260820)
    parser.add_argument('--out-json', type=Path,
                        default=Path('Experiments/constraint_satisfaction_results.json'))
    parser.add_argument('--out-csv', type=Path,
                        default=Path('Experiments/constraint_satisfaction_results.csv'))
    parser.add_argument('--out-tex', type=Path,
                        default=Path('Experiments/constraint_satisfaction_table.tex'))
    return parser.parse_args()


def summary_features(sequences):
    sequences = np.asarray(sequences, dtype=np.float64)
    differences = np.diff(sequences, axis=1)
    centered = sequences - sequences.mean(axis=1, keepdims=True)
    denominator = np.sum(centered * centered, axis=1) + 1e-8
    autocorrelations = [
        np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / denominator
        for lag in (1, 2, 4, 8)
    ]
    quantiles = np.quantile(sequences, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
    midpoint = sequences.shape[1] // 2
    return np.column_stack([
        sequences.mean(axis=1),
        sequences.std(axis=1),
        sequences.min(axis=1),
        sequences.max(axis=1),
        quantiles,
        np.abs(differences).mean(axis=1),
        differences.std(axis=1),
        sequences[:, :midpoint].mean(axis=1),
        sequences[:, midpoint:].mean(axis=1),
        *autocorrelations,
    ])


def sample_real_features(path, pool_size, rng):
    data = np.load(path, mmap_mode='r')
    if data.ndim != 3:
        raise ValueError(f'Expected [windows, time, variables] at {path}, got {data.shape}')
    windows, _, variables = data.shape
    count = min(pool_size, windows * variables)
    flat_ids = rng.choice(windows * variables, count, replace=False)
    sequences = np.stack([
        data[index % windows, :, index // windows] for index in flat_ids
    ])
    return summary_features(sequences)


def sample_fake_features(path, pool_size, rng):
    data = np.load(path, mmap_mode='r')
    if data.ndim != 3 or data.shape[-1] != 1:
        raise ValueError(f'Expected [samples, time, 1] at {path}, got {data.shape}')
    count = min(pool_size, data.shape[0])
    ids = rng.choice(data.shape[0], count, replace=False)
    return summary_features(np.asarray(data[ids, :, 0]))


def frechet_distance(features_a, features_b):
    mean_delta = features_a.mean(axis=0) - features_b.mean(axis=0)
    cov_a = np.cov(features_a, rowvar=False)
    cov_b = np.cov(features_b, rowvar=False)
    cov_mean = sqrtm(cov_a.dot(cov_b))
    if not np.isfinite(cov_mean).all():
        offset = np.eye(cov_a.shape[0]) * 1e-6
        cov_mean = sqrtm((cov_a + offset).dot(cov_b + offset))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    distance = mean_delta.dot(mean_delta) + np.trace(cov_a + cov_b - 2.0 * cov_mean)
    return max(float(distance), 0.0)


def bootstrap_scores(fake, target, contrast, repeats, size, rng):
    scores, target_distances, contrast_distances = [], [], []
    for _ in range(repeats):
        fake_sample = fake[rng.choice(len(fake), size, replace=True)]
        target_sample = target[rng.choice(len(target), size, replace=True)]
        contrast_sample = contrast[rng.choice(len(contrast), size, replace=True)]
        target_distance = frechet_distance(fake_sample, target_sample)
        contrast_distance = frechet_distance(fake_sample, contrast_sample)
        denominator = target_distance + contrast_distance
        score = 0.5 if denominator <= 1e-12 else contrast_distance / denominator
        scores.append(score)
        target_distances.append(target_distance)
        contrast_distances.append(contrast_distance)
    return np.asarray(scores), np.asarray(target_distances), np.asarray(contrast_distances)


def load_feature_pools(args, dataset, rng):
    real, fake = {}, {}
    for scenario in args.scenarios:
        real_stem = REAL_STEMS.get(scenario, scenario)
        real_path = args.root / 'samples' / (
            f'{real_stem}_{dataset}_norm_truth_{args.length}_train.npy'
        )
        fake_path = args.root / (
            f'ddpm_fake_{scenario}_{dataset}_milestone_{args.milestone}_'
            f'mask0.0_len{args.length}.npy'
        )
        if not real_path.exists() or not fake_path.exists():
            missing = real_path if not real_path.exists() else fake_path
            raise FileNotFoundError(missing)
        real[scenario] = sample_real_features(real_path, args.pool_size, rng)
        fake[scenario] = sample_fake_features(fake_path, args.pool_size, rng)
    return real, fake


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path, summary, family_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    column_spec = 'l' + 'c' * (len(family_names) + 1)
    lines = [
        rf'\begin{{tabular}}{{{column_spec}}}',
        r'\toprule',
        'Dataset & ' + ' & '.join((*family_names, 'Overall')) + r' \\',
        r'\midrule',
    ]
    for dataset, values in summary.items():
        cells = [dataset.upper() if dataset == 'etth' else dataset.title()]
        for family in (*family_names, 'Overall'):
            result = values[family]
            cells.append(f"{100 * result['mean']:.1f}")
        lines.append(' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    path.write_text('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    if args.bootstrap_repeats < 2 or args.bootstrap_size < 2 or args.pool_size < 2:
        raise ValueError('Bootstrap repeats, bootstrap size, and pool size must be at least 2.')

    selected = tuple(args.scenarios)
    for scenario in selected:
        if scenario != 'volatile' and CONTRASTS[scenario] not in selected:
            raise ValueError(
                f'Scenario {scenario} requires contrast {CONTRASTS[scenario]} '
                f'to be included in --scenarios.'
            )
    if 'volatile' in selected and len(selected) < 2:
        raise ValueError('Volatile SPS requires at least one non-volatile scenario.')
    active_families = {
        family: scenarios
        for family, scenarios in FAMILIES.items()
        if all(name in selected for name in scenarios)
    }

    rng = np.random.RandomState(args.seed)
    rows = []
    summary = {}
    score_samples = {}

    for dataset in args.datasets:
        real, fake = load_feature_pools(args, dataset, rng)
        score_samples[dataset] = {}
        for scenario in selected:
            if scenario == 'volatile':
                nonvolatile = [name for name in selected if name != 'volatile']
                per_scenario = max(2, args.pool_size // len(nonvolatile))
                contrast = np.vstack([
                    real[name][rng.choice(len(real[name]), per_scenario, replace=True)]
                    for name in nonvolatile
                ])
            else:
                contrast = real[CONTRASTS[scenario]]

            sample_size = min(
                args.bootstrap_size, len(fake[scenario]), len(real[scenario]), len(contrast)
            )
            scores, target_distances, contrast_distances = bootstrap_scores(
                fake[scenario], real[scenario], contrast,
                args.bootstrap_repeats, sample_size, rng,
            )
            score_samples[dataset][scenario] = scores
            low, high = np.percentile(scores, [2.5, 97.5])
            row = {
                'dataset': dataset,
                'scenario': scenario,
                'contrast': CONTRASTS[scenario],
                'target_distance_mean': float(target_distances.mean()),
                'contrast_distance_mean': float(contrast_distances.mean()),
                'sps_mean': float(scores.mean()),
                'sps_ci_low': float(low),
                'sps_ci_high': float(high),
                'target_closer': bool(scores.mean() > 0.5),
            }
            rows.append(row)
            print(
                f"{dataset:>6} {scenario:>13}: SPS={100 * scores.mean():5.1f}% "
                f"[{100 * low:5.1f}, {100 * high:5.1f}] "
                f"target_closer={row['target_closer']}"
            )

        summary[dataset] = {}
        for family, scenarios in active_families.items():
            family_samples = np.mean(
                np.vstack([score_samples[dataset][name] for name in scenarios]), axis=0
            )
            low, high = np.percentile(family_samples, [2.5, 97.5])
            summary[dataset][family] = {
                'mean': float(family_samples.mean()),
                'ci_low': float(low),
                'ci_high': float(high),
            }
        overall_samples = np.mean(
            np.vstack([score_samples[dataset][name] for name in selected]), axis=0
        )
        low, high = np.percentile(overall_samples, [2.5, 97.5])
        summary[dataset]['Overall'] = {
            'mean': float(overall_samples.mean()),
            'ci_low': float(low),
            'ci_high': float(high),
        }

    payload = {
        'metric': {
            'name': 'Scenario Preference Score (SPS)',
            'definition': 'd_contrast / (d_target + d_contrast); higher is better; 0.5 is neutral',
            'distance': 'Frechet distance over transparent summary features',
            'uncertainty': '95% sample-bootstrap interval; not independent training seeds',
        },
        'settings': {
            'root': str(args.root),
            'datasets': args.datasets,
            'scenarios': list(selected),
            'milestone': args.milestone,
            'length': args.length,
            'pool_size': args.pool_size,
            'bootstrap_size': args.bootstrap_size,
            'bootstrap_repeats': args.bootstrap_repeats,
            'seed': args.seed,
        },
        'rows': rows,
        'summary': summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + '\n')
    write_csv(args.out_csv, rows)
    write_latex(args.out_tex, summary, tuple(active_families))

    print('\nFamily-level SPS (%)')
    for dataset, values in summary.items():
        formatted = ', '.join(
            f"{family}={100 * result['mean']:.1f}"
            for family, result in values.items()
        )
        print(f'{dataset}: {formatted}')
    print(f'Wrote {args.out_json}, {args.out_csv}, and {args.out_tex}')


if __name__ == '__main__':
    main()
