#!/usr/bin/env python
"""
evaluate.py — Compute C-FID, Discriminative, Predictive scores
===============================================================
Usage:
    python evaluate.py --ori <ori.npy> --fake <fake.npy> --tag "zeroshot_ettep"

Can be called from any project dir (mix_gpt2 or mix_diffts_glab)
as long as sys.path includes the project root.
"""
import os, sys, json, argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori',  required=True, help='Path to ori norm_truth npy')
    parser.add_argument('--fake', required=True, help='Path to fake npy')
    parser.add_argument('--tag',  default='', help='Label for this evaluation')
    parser.add_argument('--iters', type=int, default=5, help='Number of iterations')
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=['cfid', 'discriminative', 'predictive'],
        default=['cfid', 'discriminative', 'predictive'],
        help='Metric subset to run',
    )
    parser.add_argument(
        '--sample-seed',
        type=int,
        default=20260820,
        help='Seed used only for deterministic real/fake count balancing',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='Maximum equal-count sample pool; 0 uses all available samples',
    )
    parser.add_argument(
        '--cfid-device',
        default=None,
        help='PyTorch device for Context-FID, such as cuda:0 or cpu (default: auto)',
    )
    parser.add_argument('--out', default=None, help='Output JSON path (optional)')
    args = parser.parse_args()

    ori_data = np.load(args.ori)
    fake_data = np.load(args.fake)
    print(f'[{args.tag}] ori={ori_data.shape}  fake={fake_data.shape}')

    # Channel-independent evaluation: pool each variable trajectory as one sequence.
    def flatten_channelwise(data):
        if data.ndim == 2:
            data = data[:, :, None]
        if data.ndim != 3:
            raise ValueError(f'Expected a 2D/3D sequence array, got shape {data.shape}')
        batch, length, channels = data.shape
        return data.transpose(2, 0, 1).reshape(batch * channels, length, 1)

    ori_flat = flatten_channelwise(ori_data)
    fake_flat = flatten_channelwise(fake_data)
    if ori_flat.shape[1:] != fake_flat.shape[1:]:
        raise ValueError(
            f'Real/fake sequence shapes differ after channel pooling: '
            f'{ori_flat.shape[1:]} vs {fake_flat.shape[1:]}'
        )

    # Compare equal sample counts.  Subsampling is deterministic and independent
    # of the stochastic metric-estimation repeats below.
    compare_count = min(len(ori_flat), len(fake_flat))
    if args.max_samples > 0:
        compare_count = min(compare_count, args.max_samples)
    if compare_count == 0:
        raise ValueError('Cannot evaluate an empty real or synthetic array.')
    rng = np.random.RandomState(args.sample_seed)
    if len(ori_flat) > compare_count:
        ori_flat = ori_flat[rng.choice(len(ori_flat), compare_count, replace=False)]
    if len(fake_flat) > compare_count:
        fake_flat = fake_flat[rng.choice(len(fake_flat), compare_count, replace=False)]
    print(
        f'  After channel pooling/balancing: ori={ori_flat.shape} '
        f'fake={fake_flat.shape} seed={args.sample_seed}'
    )

    results = {
        'tag': args.tag,
        'ori_shape': list(ori_data.shape),
        'fake_shape': list(fake_data.shape),
        'evaluated_count': int(compare_count),
        'sample_seed': int(args.sample_seed),
    }

    # ---- C-FID (PyTorch) ----
    try:
        if 'cfid' not in args.metrics:
            raise RuntimeError('disabled by --metrics')
        from Utils.context_fid import Context_FID
        scores = []
        for i in range(args.iters):
            s = Context_FID(ori_flat, fake_flat, device=args.cfid_device)
            scores.append(s)
            print(f'  C-FID iter {i}: {s:.4f}')
        mean_s, std_s = np.mean(scores), np.std(scores)
        print(f'  C-FID = {mean_s:.4f} +/- {std_s:.4f}')
        results['cfid_mean'] = float(mean_s)
        results['cfid_std'] = float(std_s)
    except Exception as e:
        label = 'disabled' if str(e) == 'disabled by --metrics' else f'failed: {e}'
        print(f'  [SKIP] C-FID {label}')

    # ---- Discriminative Score (TensorFlow) ----
    try:
        if 'discriminative' not in args.metrics:
            raise RuntimeError('disabled by --metrics')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        from Utils.discriminative_metric import discriminative_score_metrics
        scores = []
        for i in range(args.iters):
            s, fa, ra = discriminative_score_metrics(ori_flat, fake_flat)
            scores.append(s)
            print(f'  Discri iter {i}: {s:.4f}  (fake_acc={fa:.3f}, real_acc={ra:.3f})')
        mean_s, std_s = np.mean(scores), np.std(scores)
        print(f'  Discriminative = {mean_s:.4f} +/- {std_s:.4f}')
        results['discri_mean'] = float(mean_s)
        results['discri_std'] = float(std_s)
    except Exception as e:
        label = 'disabled' if str(e) == 'disabled by --metrics' else f'failed: {e}'
        print(f'  [SKIP] Discriminative {label}')

    # ---- Predictive Score (TensorFlow) ----
    try:
        if 'predictive' not in args.metrics:
            raise RuntimeError('disabled by --metrics')
        from Utils.predictive_metric import predictive_score_metrics
        scores = []
        for i in range(args.iters):
            s = predictive_score_metrics(ori_flat, fake_flat)
            scores.append(s)
            print(f'  Predic iter {i}: {s:.4f}')
        mean_s, std_s = np.mean(scores), np.std(scores)
        print(f'  Predictive = {mean_s:.4f} +/- {std_s:.4f}')
        results['predic_mean'] = float(mean_s)
        results['predic_std'] = float(std_s)
    except Exception as e:
        label = 'disabled' if str(e) == 'disabled by --metrics' else f'failed: {e}'
        print(f'  [SKIP] Predictive {label}')

    # ---- Save results ----
    out_path = args.out or f'eval_{args.tag}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Results saved to {out_path}')
    print('=' * 60)

if __name__ == '__main__':
    main()
