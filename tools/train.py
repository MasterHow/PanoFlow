import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from opticalflow.api import init_model, manage_data
from opticalflow.api.evaluate import (sequence_loss, validate_chairs, validate_flow360)
from opticalflow.utils.logger import Logger

# sum events to one message
SUM_FREQ = 100

# validation frequency
VAL_FREQ = 10000


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate the model')
    parser.add_argument(
        '--train', help='True or False', default=True, choices=[True, False])
    parser.add_argument(
        '--model',
        help='The model used to train and inference',
        default='PanoFlow(CSFlow)',
        choices=['CSFlow', 'RAFT', 'PanoFlow(CSFlow)', 'PanoFlow(RAFT)'])
    parser.add_argument(
        '--name', help='name your experiment', default='PanoFlow(CSFlow)-test')
    parser.add_argument(
        '--restore_ckpt',
        help='Restored checkpoint you are using/path or None',
        default=None)
    parser.add_argument(
        '--start_step',
        type=int,
        help='Start of train steps, used for RESUME',
        default=0)
    parser.add_argument(
        '--num_steps',
        type=int,
        help='Total number of train steps',
        default=100)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument(
        '--dataset',
        help='The data use to train',
        default='Chairs',
        choices=['Chairs', 'Things', 'Flow360'])
    parser.add_argument(
        '--image_size',
        type=int,
        nargs='+',
        help='Cropped img size used to train',
        default=[384, 512])
    parser.add_argument(
        '--data_root',
        help='Root of the current training datasets')
    parser.add_argument(
        '--validation',
        type=str,
        nargs='+',
        help='The dataset used to validate')
    parser.add_argument(
        '--train_Flow360_root',
        help='Root of the Flow360 datasets')
    parser.add_argument(
        '--val_Chairs_root',
        help='Root of the Chairs validation datasets')
    parser.add_argument(
        '--val_Flow360_root',
        help='Root of the current datasets')
    parser.add_argument('--DEVICE', help='The using device', default='cuda')
    parser.add_argument(
        '--lr', type=float, help='Learning rate', default=0.00002)
    parser.add_argument(
        '--wdecay',
        type=float,
        help='Decay rate of learning rate ',
        default=.00005)
    parser.add_argument(
        '--gamma',
        type=float,
        help='exponential weighting the loss',
        default=0.8)
    parser.add_argument(
        '--change_gpu',
        help='train on cuda device but not cuda:0',
        action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument(
        '--iters',
        type=int,
        help='Iterations of GRU unit, 12 for train',
        default=12)
    parser.add_argument(
        '--eval_iters',
        type=int,
        help='Iterations of GRU unit when eval',
        default=24)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()
    return args


def fetch_optimizer(args, model, data_length):
    """Create the optimizer and learning rate scheduler, from RAFT."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wdecay,
        eps=args.epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='linear')

    return optimizer, scheduler


def main():
    args = parse_args()

    # random seed
    if args.model == 'CSFlow':
        torch.manual_seed(1234)
        np.random.seed(1234)

    batch_size = int(args.batch_size)

    # Prepare dataloader
    training_data = manage_data.fetch_training_data(args)
    dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    # Init model
    model = init_model(args)

    # Wrap optimizer
    optimizer, scheduler = fetch_optimizer(args, model, len(training_data))
    scaler = torch.cuda.amp.GradScaler()
    logger = Logger(model, scheduler, SUM_FREQ, args.start_step)

    # Wrap training loop

    total_steps = 0
    if args.start_step > 0:
        # For resume training
        should_keep_training = True
        while should_keep_training:
            scheduler.step()
            total_steps += 1

            if total_steps > args.start_step:
                break

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(dataloader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [
                x.cuda(args.gpus[0]) for x in data_blob
            ]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 +
                          stdv * torch.randn(*image1.shape).cuda()).clamp(
                              0.0, 255.0)
                image2 = (image2 +
                          stdv * torch.randn(*image2.shape).cuda()).clamp(
                              0.0, 255.0)

            # zip image
            image_pair = torch.stack((image1, image2))
            flow_predictions = model(image_pair)

            loss, metrics = sequence_loss(flow_predictions, flow, valid,
                                          args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = './checkpoints/%d_%s.pth' % (total_steps + 1,
                                                    args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'Chairs':
                        if args.change_gpu:
                            results.update(
                                validate_chairs(model,
                                                args.val_Chairs_root,
                                                args.gpus))
                        else:
                            results.update(
                                validate_chairs(model.module,
                                                args.val_Chairs_root))
                    elif val_dataset == 'Flow360':
                        if args.change_gpu:
                            results.update(
                                validate_flow360(model, args.val_Flow360_root, args.gpus))
                        else:
                            results.update(
                                validate_flow360(model.module, args.val_Flow360_root))

                logger.write_dict(results)

            model.train()
            if args.dataset != 'Chairs':
                try:
                    model.module.freeze_bn()
                except (Exception):
                    try:
                        model.freeze_bn()
                    except:
                        for m in model.modules():
                            if isinstance(m, torch.nn.BatchNorm2d):
                                m.eval()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = './checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    print('Looks nice! Wish you a good luck! =)')


if __name__ == '__main__':
    main()
