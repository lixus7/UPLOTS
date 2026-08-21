import os
import sys
import time
import math
import torch
import numpy as np
import random
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, ins, dataloader, train_batches, max_train_batches,logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['save_cycle'] * args.milestone
        # print('self.train_num_steps ',self.train_num_steps)
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
#         self.dl = cycle(dataloader['dataloader'])

        self.ins = ins
        self.train_loaders = dataloader
        self.train_batches = train_batches
        self.max_train_batches=max_train_batches
        # print('train_batches is ',train_batches)
        self.step = 0
        self.milestone = 0
        self.current_epoch = 0
        self.args = args
        self.logger = logger

        self.loader_batch_counts = [len(dl) for dl in self.train_loaders]
        self.loss_window = int(args.loss_window)
        self.loss_histories = [deque(maxlen=self.loss_window) for _ in self.train_loaders]
        self.calm_fixed_weights = None

        self.results_folder = Path('Checkpoints_'+ args.name + f'_{model.seq_length}'+ f'_maskrate{args.mask_rate}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, epoch=None, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'epoch': epoch if epoch is not None else getattr(self, 'current_epoch', 0),
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'loss_histories': [list(history) for history in self.loss_histories],
            'calm_fixed_weights': self.calm_fixed_weights,
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        ckpt_path = self.results_folder / f'checkpoint-{milestone}.pt'
        # 如果文件不存在，给出提示并跳过加载，返回当前 epoch（默认为 0）
        if not ckpt_path.exists():
            if self.logger is not None:
                self.logger.log_info(f'Checkpoint file not found: {ckpt_path}, skip loading.')
            return getattr(self, 'current_epoch', 0)

        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(ckpt_path)))
        device = self.device
        data = torch.load(str(ckpt_path), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        saved_histories = data.get('loss_histories')
        if saved_histories is not None and len(saved_histories) == len(self.loss_histories):
            self.loss_histories = [
                deque(history, maxlen=self.loss_window) for history in saved_histories
            ]
        self.calm_fixed_weights = data.get('calm_fixed_weights')
        self.milestone = milestone
        # 恢复 epoch 信息
        self.current_epoch = data.get('epoch', milestone)
        return self.current_epoch

    def find_latest_checkpoint(self):
        """查找最新的 checkpoint，返回 milestone 和 epoch"""
        if not self.results_folder.exists():
            return None, 0

        checkpoints = list(self.results_folder.glob('checkpoint-*.pt'))
        if not checkpoints:
            return None, 0

        # 提取 milestone 并排序
        milestones = []
        for cp in checkpoints:
            try:
                milestone = int(cp.stem.split('-')[1])
                milestones.append((milestone, cp))
            except:
                continue

        if not milestones:
            return None, 0

        # 找到最大的 milestone
        latest_milestone, latest_cp = max(milestones, key=lambda x: x[0])

        # 读取 epoch 信息
        try:
            data = torch.load(str(latest_cp), map_location='cpu')
            epoch = data.get('epoch', latest_milestone)
            return latest_milestone, epoch
        except:
            return latest_milestone, latest_milestone



    def _rolling_losses(self):
        return np.asarray([
            float(np.mean(history)) if history else 1.0
            for history in self.loss_histories
        ], dtype=np.float64)

    def _calm_weights(self):
        """Map higher rolling losses to smaller curriculum weights in [s, 1]."""
        num_datasets = len(self.train_loaders)
        if self.args.disable_calm or num_datasets == 1:
            return {idx: 1.0 for idx in range(num_datasets)}

        rolling = self._rolling_losses()
        loss_max, loss_min = float(rolling.max()), float(rolling.min())
        if not np.isfinite(loss_max - loss_min) or loss_max - loss_min <= 1e-12:
            return {idx: 1.0 for idx in range(num_datasets)}

        min_weight = float(self.args.calm_min_weight)
        weights = min_weight + (loss_max - rolling) / (loss_max - loss_min) * (1.0 - min_weight)
        return {idx: float(weight) for idx, weight in enumerate(weights)}

    def _rlds_probabilities(self):
        """Return loss-proportional dataset probabilities for the next micro-batch."""
        num_datasets = len(self.train_loaders)
        if self.args.disable_rlds or num_datasets == 1:
            return np.full(num_datasets, 1.0 / num_datasets, dtype=np.float64)

        rolling = np.clip(self._rolling_losses(), 1e-12, None)
        temperature = float(self.args.rlds_temperature)
        scaled = np.power(rolling, 1.0 / temperature)
        if not np.all(np.isfinite(scaled)) or scaled.sum() <= 0:
            return np.full(num_datasets, 1.0 / num_datasets, dtype=np.float64)
        return scaled / scaled.sum()

    @staticmethod
    def _next_cycled_batch(loader, iterator):
        """Draw a batch and restart an exhausted loader so RLDS can truly resample it."""
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            try:
                return next(iterator), iterator
            except StopIteration as exc:
                raise RuntimeError('Encountered an empty training dataloader.') from exc

    def train(self, start_epoch=None):
        device = self.device
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        weak_dataset_idx = self.args.weak_idx
        alpha = self.args.alpha

        # 确定起始 epoch
        if start_epoch is None:
            start_epoch = getattr(self, 'current_epoch', 0)

        # 确保训练到指定的 epoch 数
        target_epoch = self.args.epoch
        if start_epoch >= target_epoch:
            if self.logger is not None:
                self.logger.log_info(f'Already reached target epoch {target_epoch}, current epoch: {start_epoch}')
            return

        if self.logger is not None:
            self.logger.log_info(f'Resuming training from epoch {start_epoch} to {target_epoch}')

        num_datasets = len(self.train_loaders)
        steps_per_epoch = math.ceil(self.train_batches / self.gradient_accumulate_every)
        total_steps = steps_per_epoch * (target_epoch - start_epoch)
        with tqdm(total=total_steps) as pbar:
            for e in range(start_epoch, target_epoch):
                self.current_epoch = e

                print('##############')
                print('Current Epoch ', e)
                print('##############')
                if e < self.args.calm_warmup_epochs:
                    weight_by_idx = self._calm_weights()
                else:
                    if self.calm_fixed_weights is None:
                        self.calm_fixed_weights = self._calm_weights()
                    weight_by_idx = self.calm_fixed_weights
                if self.args.disable_calm or e >= self.args.calm_active_epochs:
                    weight_by_idx = {idx: 1.0 for idx in range(num_datasets)}

                iterators = [iter(loader) for loader in self.train_loaders]
                batch_cnt = [0] * num_datasets
                micro_batches_done = 0

                # Without RLDS, preserve the original per-loader batch counts exactly.
                static_schedule = None
                if self.args.disable_rlds:
                    static_schedule = [
                        idx
                        for idx, count in enumerate(self.loader_batch_counts)
                        for _ in range(count)
                    ]
                    epoch_rng = np.random.RandomState(self.args.seed + e)
                    epoch_rng.shuffle(static_schedule)

                while micro_batches_done < self.train_batches:
                    accumulate_now = min(
                        self.gradient_accumulate_every,
                        self.train_batches - micro_batches_done,
                    )
                    raw_loss_sum = 0.0
                    weighted_loss_sum = 0.0
                    self.opt.zero_grad()

                    for _ in range(accumulate_now):
                        if static_schedule is None:
                            probs = self._rlds_probabilities()
                            idx = int(np.random.choice(num_datasets, p=probs))
                        else:
                            idx = static_schedule[micro_batches_done]

                        data, iterators[idx] = self._next_cycled_batch(
                            self.train_loaders[idx], iterators[idx]
                        )
                        data = data.to(device, non_blocking=True)

                        # Shared channel-independent model: each variable becomes one sequence.
                        b, t, n = data.shape
                        data = data.permute(0, 2, 1).reshape(b * n, t, 1)
                        mask = (torch.rand((b * n, t, 1), device=device) >= self.args.mask_rate).float()
                        model_input = data.masked_fill(mask == 0, 0)

                        raw_loss = self.model(
                            self.ins[idx], model_input, mask=mask, target=model_input
                        )
                        raw_value = float(raw_loss.detach().item())
                        self.loss_histories[idx].append(raw_value)

                        curriculum_weight = weight_by_idx.get(idx, 1.0)
                        manual_weight = alpha if idx == weak_dataset_idx else 1.0
                        weighted_loss = raw_loss * curriculum_weight * manual_weight
                        (weighted_loss / accumulate_now).backward()

                        raw_loss_sum += raw_value
                        weighted_loss_sum += float(weighted_loss.detach().item())
                        batch_cnt[idx] += 1
                        micro_batches_done += 1

                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    scheduler_loss = weighted_loss_sum / accumulate_now
                    self.sch.step(scheduler_loss)
                    self.step += 1
                    self.ema.update()

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(
                            tag='train/raw_loss',
                            scalar_value=raw_loss_sum / accumulate_now,
                            global_step=self.step,
                        )
                        self.logger.add_scalar(
                            tag='train/weighted_loss',
                            scalar_value=scheduler_loss,
                            global_step=self.step,
                        )

                    pbar.update(1)
                    pbar.set_description(
                        f'Step: {self.step}, Raw: {raw_loss_sum / accumulate_now:.6f}, '
                        f'Weighted: {scheduler_loss:.6f}'
                    )

                if e + 1 == self.args.calm_warmup_epochs:
                    self.calm_fixed_weights = self._calm_weights()

                if self.logger is not None:
                    self.logger.log_info(
                        'Epoch {} sampling counts: {}; CALM weights: {}'.format(
                            e + 1,
                            batch_cnt,
                            [round(weight_by_idx[idx], 6) for idx in range(num_datasets)],
                        ),
                        check_primary=False,
                    )

                self.milestone = e + 1
                if ((e + 1) % self.args.checkpoint_every == 0) or (e + 1 == target_epoch):
                    with torch.no_grad():
                        self.save(self.milestone, epoch=e + 1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, instruct, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        sample_chunks = []
        generated = 0
        while generated < num:
            current_batch = min(size_every, num - generated)
            sample = self.ema.ema_model.generate_mts(instruct, batch_size=current_batch)
            sample_chunks.append(sample.detach().cpu().numpy())
            generated += current_batch
            torch.cuda.empty_cache()

        samples = np.concatenate(sample_chunks, axis=0) if sample_chunks else np.empty(
            [0, shape[0], shape[1]]
        )

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])

        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples
