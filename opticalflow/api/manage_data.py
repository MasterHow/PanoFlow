import os.path as osp

import cv2

from opticalflow.core.dataset import KITTIDemoManager
from opticalflow.dataset import (FlyingChairs, FlyingThings3D, Flow360)


def load_data(args):
    return KITTIDemoManager.load_images(args.img_prefix, args)


def create_dataloader(data, args):
    return KITTIDemoManager.create_dataloader(data, args)


def output_data(imgs, output_dir):
    file_path = osp.join(output_dir, 'demo.jpg')
    cv2.imwrite(file_path, imgs)


def fetch_training_data(args):
    """Create the data loader for the corresponding trainign set."""

    if args.dataset == 'Chairs':
        if args.model == 'PanoFlow(CSFlow)' or args.model == 'PanoFlow(RAFT)':
            do_distort = True
        else:
            do_distort = False
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.1,
            'max_scale': 1.0,
            'do_flip': True,
            'do_distort': do_distort
        }
        training_data = FlyingChairs(
            aug_params, split='training', root=args.data_root)

    elif args.dataset == 'Things':
        if args.model == 'PanoFlow(CSFlow)' or args.model == 'PanoFlow(RAFT)':
            do_distort = True
        else:
            do_distort = False
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.4,
            'max_scale': 0.8,
            'do_flip': True,
            'do_distort': do_distort
        }
        clean_dataset = FlyingThings3D(
            aug_params, dstype='frames_cleanpass', root=args.data_root)
        final_dataset = FlyingThings3D(
            aug_params, dstype='frames_finalpass', root=args.data_root)
        training_data = clean_dataset + final_dataset

    elif args.dataset == 'Flow360':
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.2,
            'max_scale': 0.6,
            'do_flip': True,
            'do_distort': False
        }
        sunny = Flow360(
            aug_params,
            split='train',
            root=args.train_Flow360_root,
            dstype='sunny')
        cloud = Flow360(
            aug_params,
            split='train',
            root=args.train_Flow360_root,
            dstype='cloud')
        rain = Flow360(
            aug_params,
            split='train',
            root=args.train_Flow360_root,
            dstype='rain')
        fog = Flow360(
            aug_params,
            split='train',
            root=args.train_Flow360_root,
            dstype='fog')
        training_data = sunny + cloud + rain + fog


    print('Training with %d image pairs' % len(training_data))
    return training_data
