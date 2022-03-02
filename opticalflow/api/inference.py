from collections import OrderedDict

import torch

from opticalflow.utils.utils import (fill_order_keys, fix_order_keys,
                                     fix_read_order_keys)


def inference(model, x, args):
    if args.model == 'RAFT':
        if args.train:
            model = torch.nn.DataParallel(model._model)
            try:
                model.load_state_dict(torch.load(args.checkpoint))
            except (Exception):
                d1 = torch.load(args.checkpoint)
                d2 = OrderedDict([(fix_order_keys(k, 6), v)
                                  for k, v in d1.items()])
                model.load_state_dict(d2)
            model = model.module
            model.to(args.DEVICE)
            model.eval()
            with torch.no_grad():
                return model(x, test_mode=True)
        else:  # fix model load bug when test
            model = torch.nn.DataParallel(model._model)
            try:
                model.load_state_dict(
                    torch.load(args.checkpoint, map_location='cpu'))
            except (Exception):
                try:
                    d1 = torch.load(args.checkpoint, map_location='cpu')
                    d2 = OrderedDict([(fix_order_keys(k, 6), v)
                                      for k, v in d1.items()])
                    model.load_state_dict(d2)
                except (Exception):
                    d1 = torch.load(args.checkpoint, map_location='cpu')
                    d2 = OrderedDict([(fill_order_keys(
                        fix_read_order_keys(k, 7),
                        fill_value='module.',
                        fill_position=0), v) for k, v in d1.items()])
                    model.load_state_dict(d2)
            model = model.module
            model.to(args.DEVICE)
            model.eval()
            with torch.no_grad():
                return model(x, test_mode=True)
    else:
        with torch.no_grad():
            return model(x)
