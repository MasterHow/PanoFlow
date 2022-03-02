import argparse

from opticalflow.api import init_model
from opticalflow.api.evaluate import validate_flow360, validate_flow360_cfe, validate_omni_cfe, validate_omni


def parse_args():
    parser = argparse.ArgumentParser(description='Test and evaluate the model')
    parser.add_argument(
        '--model',
        help='The model use to inference',
        default='CSFlow',
        choices=['CSFlow', 'RAFT', 'PanoFlow(CSFlow)', 'PanoFlow(RAFT)'])
    parser.add_argument(
        '--CFE',
        help='inference under CFE framework, details in paper',
        action='store_true')
    parser.add_argument(
        '--restore_ckpt',
        help="Restored checkpoint you are using/path or None",
        default='./checkpoints/raft-things.pth')
    parser.add_argument(
        '--iters',
        type=int,
        help='Iterations of GRU unit when train',
        default=20)
    parser.add_argument(
        '--eval_iters',
        type=int,
        help='Iterations of GRU unit when eval',
        default=12)
    parser.add_argument(
        '--train', help='True or False', default=True, choices=[True, False])
    parser.add_argument(
        '--eval',
        default=True,
        help='Whether eval or test demo',
        choices=[True, False])
    parser.add_argument(
        '--dataset',
        help='The data use to train',
        default='Things')
    parser.add_argument(
        '--val_Flow360_root',
        help='Root of the current datasets')
    parser.add_argument(
        '--val_Omni_root',
        help='Root of the current datasets')
    parser.add_argument(
        '--validation',
        type=str,
        nargs='+',
        default=[],
        help='The dataset used to validate RAFT')
    parser.add_argument(
        '--change_gpu',
        help='train on cuda device but not cuda:0',
        action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--DEVICE', help='The using device', default='cuda')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(args)

    results = {}
    if args.CFE:
        for val_dataset in args.validation:
            if val_dataset == 'Flow360':
                if args.change_gpu:
                    results.update(
                        validate_flow360_cfe(model, args.val_Flow360_root, args.gpus))
                else:
                    results.update(
                        validate_flow360_cfe(model.module, args.val_Flow360_root))
            if val_dataset == 'Omni':
                if args.change_gpu:
                    results.update(
                        validate_omni_cfe(model, args.val_Omni_root, args.gpus))
                else:
                    results.update(
                        validate_omni_cfe(model.module, args.val_Omni_root))
    else:
        for val_dataset in args.validation:
            if val_dataset == 'Flow360':
                if args.change_gpu:
                    results.update(
                        validate_flow360(model, args.val_Flow360_root, args.gpus))
                else:
                    results.update(
                        validate_flow360(model.module, args.val_Flow360_root))
            elif val_dataset == 'Omni':
                if args.change_gpu:
                    results.update(
                        validate_omni(model, args.val_Omni_root, args.gpus))
                else:
                    results.update(
                        validate_omni(model.module, args.val_Omni_root))


if __name__ == '__main__':
    main()
