import os
import torch
import argparse
import numpy as np
import json
from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


PROMPT_CODES = {
    # base datasets
    'etth': 'etth', 'energy': 'energy', 'pems04': 'pems04', 'pems08': 'pems08',
    # morning / evening peak
    'morning_peak_etth': 'ETTMP', 'evening_peak_etth': 'ETTEP',
    'morning_peak_energy': 'ENEMP', 'evening_peak_energy': 'ENEEP',
    'morning_peak_pems04': 'PEMS04MP', 'evening_peak_pems04': 'PEMS04EP',
    'morning_peak_pems08': 'PEMS08MP', 'evening_peak_pems08': 'PEMS08EP',
    # workday / weekend
    'workday_etth': 'ETTWKD', 'weekend_etth': 'ETTWKE',
    'workday_energy': 'ENEWKD', 'weekend_energy': 'ENEWKE',
    'workday_pems04': 'P04WKD', 'weekend_pems04': 'P04WKE',
    'workday_pems08': 'P08WKD', 'weekend_pems08': 'P08WKE',
    # high-load / low-load
    'high_load_etth': 'ETTHI', 'low_load_etth': 'ETTLO',
    'high_load_energy': 'ENEHI', 'low_load_energy': 'ENELO',
    'high_load_pems04': 'P04HI', 'low_load_pems04': 'P04LO',
    'high_load_pems08': 'P08HI', 'low_load_pems08': 'P08LO',
    # volatility
    'volatile_etth': 'ETTVOL', 'volatile_energy': 'ENEVOL',
    'volatile_pems04': 'P04VOL', 'volatile_pems08': 'P08VOL',
    # legacy keys
    'workdays': 'WORK', 'weekends': 'WEEKEND', 'daytime': 'DAT', 'nighttime': 'NIGHT',
}
PROMPT_IDS = {name: idx for idx, name in enumerate(PROMPT_CODES)}

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', nargs='+', type=str, default=None,
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT',
                        help='directory to save the results')
    parser.add_argument('--instruct_path', type=str, default='./Data/datasets/prompts.json',
                        help='prompt')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')

    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0,
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mask_rate', type=float, default=0.0,  help='mask rate.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--weak_idx', type=int, default=-1,
                        help='optional manual dataset weight index; -1 disables it')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seq_length', '--seq_len', dest='seq_length', type=int, default=24,
                        help='generated sequence length (legacy alias: --seq_len)')
    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')

    # multi-dataset training and backbone ablations
    parser.add_argument('--disable_calm', action='store_true',
                        help='disable Curriculum-Aware Loss Modulation')
    parser.add_argument('--disable_rlds', action='store_true',
                        help='disable Rolling-Loss Dynamic Sampling')
    parser.add_argument('--calm_warmup_epochs', type=int, default=5)
    parser.add_argument('--calm_active_epochs', type=int, default=50)
    parser.add_argument('--calm_min_weight', type=float, default=0.9)
    parser.add_argument('--loss_window', type=int, default=100)
    parser.add_argument('--rlds_temperature', type=float, default=1.0)
    parser.add_argument('--checkpoint_every', type=int, default=50)
    parser.add_argument('--backbone', choices=['gpt2', 'llama'], default='gpt2',
                        help='language backbone family (default: gpt2)')
    parser.add_argument('--backbone_name', type=str, default=None,
                        help='Hugging Face model id or local path; defaults depend on --backbone')
    parser.add_argument('--backbone_layers', type=int, default=None,
                        help='number of retained backbone layers (defaults: GPT-2=2, LLaMA=8)')
    parser.add_argument('--backbone_init', choices=['pretrained', 'random'], default='pretrained',
                        help='initialize the selected backbone from pretrained or random weights')
    parser.add_argument('--prompt_mode', choices=['text', 'learned_id', 'none'], default='text',
                        help='use text tokens, a learned scenario ID, or no prompt')
    parser.add_argument('--train_backbone', action='store_true',
                        help='fine-tune the entire backbone instead of only normalization/position parameters')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='number of generated univariate sequences; 0 matches the real channel count')
    parser.add_argument('--sample_batch_size', type=int, default=900)

    # args for modify config
    parser.add_argument('--current_ins', type=str, default=' ',
                        help='ins for inference')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if not args.config_file:
        parser.error('--config_file requires at least one configuration name')
    if args.loss_window <= 0:
        parser.error('--loss_window must be positive')
    if args.rlds_temperature <= 0:
        parser.error('--rlds_temperature must be positive')
    if not 0 < args.calm_min_weight <= 1:
        parser.error('--calm_min_weight must be in (0, 1]')
    if args.checkpoint_every <= 0:
        parser.error('--checkpoint_every must be positive')
    if args.sample_batch_size <= 0:
        parser.error('--sample_batch_size must be positive')
    if args.backbone_layers is not None and args.backbone_layers <= 0:
        parser.error('--backbone_layers must be positive')
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    configfile = args.config_file
    print('all config files: ',configfile)
    loaders = []
    train_batches = 0
    ins = []
    max_train_batches = 0
    logger = Logger(args)
    for config_item in configfile:
        if config_item not in PROMPT_CODES:
            raise KeyError(f'No prompt specification registered for config: {config_item}')
        if args.prompt_mode == 'text':
            ins.append(PROMPT_CODES[config_item])
        elif args.prompt_mode == 'learned_id':
            ins.append(PROMPT_IDS[config_item])
        else:
            ins.append(None)

        config_path = "./Config/" + config_item + ".yaml"
        config = load_yaml_config(config_path)
        config['dataloader']['train_dataset']['params']['window']=args.seq_length
        config['dataloader']['test_dataset']['params']['window']=args.seq_length
        config['model']['params']['seq_length']=args.seq_length
        config['model']['params']['d_model']=args.d_model
        config['model']['params']['backbone'] = args.backbone
        config['model']['params']['backbone_name'] = args.backbone_name
        config['model']['params']['backbone_layers'] = args.backbone_layers
        config['model']['params']['backbone_init'] = args.backbone_init
        config['model']['params']['prompt_mode'] = args.prompt_mode
        config['model']['params']['num_prompt_ids'] = len(PROMPT_IDS)
        config['model']['params']['train_backbone'] = args.train_backbone

        print('config file is', config)
        config = merge_opts_to_config(config, args.opts)

        logger.save_config(config)

        dataloader_info = build_dataloader(config, args)
        dataloader = dataloader_info['dataloader']
        loaders.append(dataloader)
        # print('len(dataloader_info) SIZE :' ,len(dataloader_info))
        train_batches += len(dataloader)
        max_train_batches = max(len(dataloader), max_train_batches)
        print('train_batches : ',len(dataloader))
        # print('dataloader  : ',dataloader_info['dataloader'])
    model = instantiate_from_config(config['model']).cuda()
    print('instruct_list: ',ins)

    print('train batches : ', train_batches)
    trainer = Trainer(config=config, args=args, model=model, ins = ins, dataloader=loaders, train_batches=train_batches, max_train_batches=max_train_batches, logger=logger)

    if args.train:
        # 自动检测最新的 checkpoint
        latest_milestone, latest_epoch = trainer.find_latest_checkpoint()

        if latest_milestone is not None and latest_milestone > 0:
            if logger is not None:
                logger.log_info(f'Found latest checkpoint: milestone={latest_milestone}, epoch={latest_epoch}')
            # 如果指定了 milestone，使用指定的；否则使用最新的
            if args.milestone == 0:
                args.milestone = latest_milestone
                restored_epoch = trainer.load(args.milestone, verbose=True)
                if logger is not None:
                    logger.log_info(f'Resuming from epoch {restored_epoch}')
            else:
                restored_epoch = trainer.load(args.milestone, verbose=True)
                if logger is not None:
                    logger.log_info(f'Loading checkpoint milestone {args.milestone}, epoch {restored_epoch}')
        elif args.milestone != 0:
            # 手动指定了 milestone，但没有找到自动检测的
            restored_epoch = trainer.load(args.milestone, verbose=True)
            if logger is not None:
                logger.log_info(f'Loading checkpoint milestone {args.milestone}, epoch {restored_epoch}')
        else:
            restored_epoch = 0
            if logger is not None:
                logger.log_info('No checkpoint found, starting from scratch')

        # 从恢复的 epoch 继续训练到指定的 epoch 数
        trainer.train(start_epoch=restored_epoch)
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        if len(configfile) != 1:
            raise ValueError('Conditional sampling expects exactly one --config_file entry.')
        test_dataloader_info = build_dataloader_cond(config, args)
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}.npy'), samples)
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        requested_samples = args.num_samples or (len(dataset) * dataset.var_num)
        print('num is: ', requested_samples)
        samples = trainer.sample(
            instruct=ins[0],
            num=requested_samples,
            size_every=args.sample_batch_size,
            shape=[dataset.window, 1],
        )
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.config_file[0]}_milestone_{args.milestone}_mask{args.mask_rate}_len{args.seq_length}.npy'), samples)

if __name__ == '__main__':
    main()
