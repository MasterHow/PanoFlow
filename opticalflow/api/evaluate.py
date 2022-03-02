import numpy as np
import torch

import opticalflow.dataset as dataset
from opticalflow.utils.utils import InputPadder

class Evaluator():

    def __init__(self, data_size: int = None):
        self._loss_list = []
        self._data_size = data_size
        self._count = 0
        self._data_completed_count = 0

    def record_result(self, y, y_gt):
        pass

    def record_loss(self, loss, current_batch_size):
        self._loss_list.append(loss)
        self._data_completed_count += current_batch_size

    def print_current_evaluation_result(self):
        self._count += 1
        loss = self._loss_list[-1]
        print(f'({self._count}) loss: {loss:>7f}  '
              f'[{self._data_completed_count:>5d}/{self._data_size:>5d}]')

    def print_all_evaluation_result(self):
        print('Looks good')


def init_evaluator(data_size=1):
    return Evaluator(data_size)


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=400):
    """Loss function defined over sequence of flow predictions, from RAFT."""

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


class Not360Exception(Exception):
    def __init__(self):
        print("Not the 360° flow ground truth! Please check if your gt is converted or trun off the CFE.")


@torch.no_grad()
def validate_chairs(model, data_root, gpus=[0]):
    """Perform evaluation on the FlyingChairs (test) split, from RAFT,
    modified."""
    model.eval()
    epe_list = []

    val_dataset = dataset.FlyingChairs(split='validation', root=data_root)
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda(gpus[0])
        image2 = image2[None].cuda(gpus[0])

        # zip image
        image_pair = torch.stack((image1, image2))

        _, flow_pr = model._model(image_pair, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print('Validation Chairs EPE: %f' % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_omni(model, data_root, gpus=[0]):
    """Peform validation using the OmniDataset"""
    model.eval()
    results = {}
    epe_any = []

    for dstype in ['CartoonTree', 'Forest', 'LowPolyModels']:
        val_dataset = dataset.OmniDataset(root=data_root, dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda(gpus[0])
            image2 = image2[None].cuda(gpus[0])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # zip image
            image_pair = torch.stack((image1, image2))

            flow_low, flow_pr = model._model(image_pair, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_any.append(epe_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)

        print('Validation Omni (%s) EPE: %f' %
              (dstype, epe))
        results[dstype] = np.mean(epe_list)

    epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_final_all)
    print('Validation Omni (all) EPE: %f' %
          (epe_final))

    return results


@torch.no_grad()
def validate_omni_cfe(model, data_root, gpus=[0]):
    """Peform validation using the OmniFlowNet Dataset, under PanoFlow Framework"""
    model.eval()
    results = {}
    epe_any = []

    for dstype in ['CartoonTree', 'Forest', 'LowPolyModels']:
        val_dataset = dataset.OmniDataset(root=data_root, dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda(gpus[0])
            image2 = image2[None].cuda(gpus[0])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # check if is 360 flow gt
            if flow_gt[0, :, :].max() > flow_gt.shape[2]//2:
                raise Not360Exception()

            # zip image
            image_pair = torch.stack((image1, image2))

            # generate fmaps
            fmap1, fmap2, cnet1 = model._model(image_pair, test_mode=True, gen_fmap=True)

            # split fmaps #
            img_A1 = fmap1[:, :, :, 0:fmap1.shape[3] // 2]
            img_B1 = fmap1[:, :, :, fmap1.shape[3] // 2:]
            img_A2 = fmap2[:, :, :, 0:fmap2.shape[3] // 2]
            img_B2 = fmap2[:, :, :, fmap2.shape[3] // 2:]

            cnet_A1 = cnet1[:, :, :, 0:fmap1.shape[3] // 2]
            cnet_B1 = cnet1[:, :, :, fmap1.shape[3] // 2:]

            # prepare fmap pairs #
            img11 = torch.cat([img_B1, img_A1], dim=3)
            img21 = torch.cat([img_B2, img_A2], dim=3)
            cnet11 = torch.cat([cnet_B1, cnet_A1], dim=3)
            img_pair_B1A1 = torch.stack((img11, img21, cnet11))

            img12 = torch.cat([img_A1, img_B1], dim=3)
            img22 = torch.cat([img_A2, img_B2], dim=3)
            cnet12 = torch.cat([cnet_A1, cnet_B1], dim=3)
            img_pair_A1B1 = torch.stack((img12, img22, cnet12))

            # flow prediction #
            # skip encoder

            _, flow_pr_B1A1 = model._model(img_pair_B1A1, test_mode=True, skip_encode=True)

            _, flow_pr_A1B1 = model._model(img_pair_A1B1, test_mode=True, skip_encode=True)

            flow_pr_A1 = flow_pr_B1A1[:, :, :, flow_pr_B1A1.shape[3] // 2:]
            flow_pr_A2 = flow_pr_A1B1[:, :, :, 0:flow_pr_A1B1.shape[3] // 2]

            flow_pr_A = torch.minimum(flow_pr_A1, flow_pr_A2)

            flow_pr_B1 = flow_pr_B1A1[:, :, :, 0:flow_pr_B1A1.shape[3] // 2]
            flow_pr_B2 = flow_pr_A1B1[:, :, :, flow_pr_A1B1.shape[3] // 2:]

            flow_pr_B = torch.minimum(flow_pr_B1, flow_pr_B2)

            # all
            flow_pr = torch.cat([flow_pr_A, flow_pr_B], dim=3)
            flow_pr[:, :, :, flow_pr.shape[3] // 2] = flow_pr[:, :, :, (flow_pr.shape[3] // 2) + 1]
            flow_pr[:, :, :, (flow_pr.shape[3] // 2) - 1] = flow_pr[:, :, :, (flow_pr.shape[3] // 2) - 2]

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_any.append(epe_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        print('Validation Omni (%s) EPE: %f' %
              (dstype, epe))
        dstype = 'Flow360' + dstype
        results[dstype] = np.mean(epe_list)

    epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_final_all)
    print('Validation Omni (all) EPE: %f' %
          (epe_final))

    return results


@torch.no_grad()
def validate_flow360(model, data_root, gpus=[0]):
    """Peform validation using the Flow360 (test) split"""
    model.eval()
    results = {}
    epe_any = []

    for dstype in ['cloud', 'fog', 'rain', 'sunny']:
        val_dataset = dataset.Flow360(
            split='test', root=data_root, dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda(gpus[0])
            image2 = image2[None].cuda(gpus[0])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # zip image
            image_pair = torch.stack((image1, image2))

            flow_low, flow_pr = model._model(image_pair, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_any.append(epe_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)

        print('Validation FLow360 (%s) EPE: %f' %
              (dstype, epe))
        dstype = 'Flow360' + dstype
        results[dstype] = np.mean(epe_list)

    epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_final_all)
    print('Validation FLow360 (all) EPE: %f' %
          (epe_final))

    return results


@torch.no_grad()
def validate_flow360_cfe(model, data_root, gpus=[0]):
    """Peform validation using the Flow360 (test) split, under PanoFlow Framework"""
    model.eval()
    results = {}
    epe_any = []

    for dstype in ['cloud', 'fog', 'rain', 'sunny']:
        val_dataset = dataset.Flow360(
            split='test', root=data_root, dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda(gpus[0])
            image2 = image2[None].cuda(gpus[0])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # check if is 360 flow gt
            if flow_gt[0, :, :].max() > flow_gt.shape[2]//2:
                raise Not360Exception()

            # zip image
            image_pair = torch.stack((image1, image2))

            # generate fmaps
            fmap1, fmap2, cnet1 = model._model(image_pair, test_mode=True, gen_fmap=True)

            # split fmaps #
            img_A1 = fmap1[:, :, :, 0:fmap1.shape[3] // 2]
            img_B1 = fmap1[:, :, :, fmap1.shape[3] // 2:]
            img_A2 = fmap2[:, :, :, 0:fmap2.shape[3] // 2]
            img_B2 = fmap2[:, :, :, fmap2.shape[3] // 2:]

            cnet_A1 = cnet1[:, :, :, 0:fmap1.shape[3] // 2]
            cnet_B1 = cnet1[:, :, :, fmap1.shape[3] // 2:]

            # prepare fmap pairs #
            img11 = torch.cat([img_B1, img_A1], dim=3)
            img21 = torch.cat([img_B2, img_A2], dim=3)
            cnet11 = torch.cat([cnet_B1, cnet_A1], dim=3)
            img_pair_B1A1 = torch.stack((img11, img21, cnet11))

            img12 = torch.cat([img_A1, img_B1], dim=3)
            img22 = torch.cat([img_A2, img_B2], dim=3)
            cnet12 = torch.cat([cnet_A1, cnet_B1], dim=3)
            img_pair_A1B1 = torch.stack((img12, img22, cnet12))

            # flow prediction #
            # skip encoder

            _, flow_pr_B1A1 = model._model(img_pair_B1A1, test_mode=True, skip_encode=True)

            _, flow_pr_A1B1 = model._model(img_pair_A1B1, test_mode=True, skip_encode=True)

            flow_pr_A1 = flow_pr_B1A1[:, :, :, flow_pr_B1A1.shape[3] // 2:]
            flow_pr_A2 = flow_pr_A1B1[:, :, :, 0:flow_pr_A1B1.shape[3] // 2]

            flow_pr_A = torch.minimum(flow_pr_A1, flow_pr_A2)

            flow_pr_B1 = flow_pr_B1A1[:, :, :, 0:flow_pr_B1A1.shape[3] // 2]
            flow_pr_B2 = flow_pr_A1B1[:, :, :, flow_pr_A1B1.shape[3] // 2:]

            flow_pr_B = torch.minimum(flow_pr_B1, flow_pr_B2)

            # all
            flow_pr = torch.cat([flow_pr_A, flow_pr_B], dim=3)
            flow_pr[:, :, :, flow_pr.shape[3] // 2] = flow_pr[:, :, :, (flow_pr.shape[3] // 2) + 1]
            flow_pr[:, :, :, (flow_pr.shape[3] // 2) - 1] = flow_pr[:, :, :, (flow_pr.shape[3] // 2) - 2]

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_any.append(epe_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        print('Validation FLow360 (%s) EPE: %f' %
              (dstype, epe))
        dstype = 'Flow360' + dstype
        results[dstype] = np.mean(epe_list)

    epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_final_all)
    print('Validation FLow360 (all) EPE: %f' %
          (epe_final))

    return results


@torch.no_grad()
def validate_flow360_cfe_double_estimate(model, data_root, gpus=[0]):
    """Peform validation using the Flow360 (test) split, under PanoFlow Framework, use double estimate setting"""
    model.eval()
    results = {}
    epe_any = []

    for dstype in ['cloud', 'fog', 'rain', 'sunny']:
        val_dataset = dataset.Flow360(
            split='test', root=data_root, dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda(gpus[0])
            image2 = image2[None].cuda(gpus[0])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # check if is 360 flow gt
            if flow_gt[0, :, :].max() > flow_gt.shape[2]//2:
                raise Not360Exception()

            # split images #
            img_A1 = image1[:, :, :, 0:image1.shape[3] // 2]
            img_B1 = image1[:, :, :, image1.shape[3] // 2:]
            img_A2 = image2[:, :, :, 0:image2.shape[3] // 2]
            img_B2 = image2[:, :, :, image2.shape[3] // 2:]

            # prepare image pairs #
            img11 = torch.cat([img_B1, img_A1], dim=3)
            img21 = torch.cat([img_B2, img_A2], dim=3)
            img_pair_B1A1 = torch.stack((img11, img21))

            img12 = torch.cat([img_A1, img_B1], dim=3)
            img22 = torch.cat([img_A2, img_B2], dim=3)
            img_pair_A1B1 = torch.stack((img12, img22))

            # double estimate
            _, flow_pr_B1A1 = model._model(img_pair_B1A1, test_mode=True)

            _, flow_pr_A1B1 = model._model(img_pair_A1B1, test_mode=True)

            flow_pr_A1 = flow_pr_B1A1[:, :, :, flow_pr_B1A1.shape[3] // 2:]
            flow_pr_A2 = flow_pr_A1B1[:, :, :, 0:flow_pr_A1B1.shape[3] // 2]

            flow_pr_A = torch.minimum(flow_pr_A1, flow_pr_A2)

            flow_pr_B1 = flow_pr_B1A1[:, :, :, 0:flow_pr_B1A1.shape[3] // 2]
            flow_pr_B2 = flow_pr_A1B1[:, :, :, flow_pr_A1B1.shape[3] // 2:]

            flow_pr_B = torch.minimum(flow_pr_B1, flow_pr_B2)

            # all
            flow_pr = torch.cat([flow_pr_A, flow_pr_B], dim=3)
            flow_pr[:, :, :, flow_pr.shape[3] // 2] = flow_pr[:, :, :, (flow_pr.shape[3] // 2) + 1]
            flow_pr[:, :, :, (flow_pr.shape[3] // 2) - 1] = flow_pr[:, :, :, (flow_pr.shape[3] // 2) - 2]

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_any.append(epe_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        print('Validation FLow360 (%s) EPE: %f' %
              (dstype, epe))
        dstype = 'Flow360' + dstype
        results[dstype] = np.mean(epe_list)

    epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_final_all)
    print('Validation FLow360 (all) EPE: %f' %
          (epe_final))

    return results


@torch.no_grad()
def validate_flow360_cfe_same_padding(model, data_root, gpus=[0]):
    """Peform validation using the Flow360 (test) split, under PanoFlow Framework with same padding"""
    model.eval()
    results = {}
    epe_any = []

    for dstype in ['cloud', 'fog', 'rain', 'sunny']:
        val_dataset = dataset.Flow360(
            split='test', root=data_root, dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda(gpus[0])
            image2 = image2[None].cuda(gpus[0])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # check if is 360 flow gt
            if flow_gt[0, :, :].max() > flow_gt.shape[2]//2:
                raise Not360Exception()

            # zip image
            image_pair = torch.stack((image1, image2))

            # generate fmaps
            fmap1, fmap2, cnet1 = model._model(image_pair, test_mode=True, gen_fmap=True)

            # split fmaps #
            img_A1 = fmap1[:, :, :, 0:fmap1.shape[3] // 2]
            img_B1 = fmap1[:, :, :, fmap1.shape[3] // 2:]
            img_A2 = fmap2[:, :, :, 0:fmap2.shape[3] // 2]
            img_B2 = fmap2[:, :, :, fmap2.shape[3] // 2:]

            cnet_A1 = cnet1[:, :, :, 0:fmap1.shape[3] // 2]
            cnet_B1 = cnet1[:, :, :, fmap1.shape[3] // 2:]

            # prepare fmap pairs #
            # section A
            img11 = torch.cat([img_A1, img_A1], dim=3)
            img21 = torch.cat([img_B2, img_A2], dim=3)
            cnet11 = torch.cat([cnet_A1, cnet_A1], dim=3)
            img_pair_A1 = torch.stack((img11, img21, cnet11))

            # section B
            img13 = torch.cat([img_B1, img_B1], dim=3)
            img23 = torch.cat([img_A2, img_B2], dim=3)
            cnet13 = torch.cat([img_B1, cnet_B1], dim=3)
            img_pair_B1 = torch.stack((img13, img23, cnet13))

            # flow prediction #
            # skip encoder

            _, flow_pr_A = model._model(img_pair_A1, test_mode=True, skip_encode=True)

            _, flow_pr_B = model._model(img_pair_B1, test_mode=True, skip_encode=True)

            flow_pr_A1 = flow_pr_A[:, :, :, flow_pr_A.shape[3] // 2:]
            flow_pr_A2 = flow_pr_A[:, :, :, 0:flow_pr_A.shape[3] // 2]

            flow_pr_A = torch.minimum(flow_pr_A1, flow_pr_A2)

            flow_pr_B1 = flow_pr_B[:, :, :, flow_pr_B.shape[3] // 2:]
            flow_pr_B2 = flow_pr_B[:, :, :, 0:flow_pr_B.shape[3] // 2]

            flow_pr_B = torch.minimum(flow_pr_B1, flow_pr_B2)

            # all
            flow_pr = torch.cat([flow_pr_A, flow_pr_B], dim=3)
            flow_pr[:, :, :, flow_pr.shape[3] // 2] = flow_pr[:, :, :, (flow_pr.shape[3] // 2) + 1]
            flow_pr[:, :, :, (flow_pr.shape[3] // 2) - 1] = flow_pr[:, :, :, (flow_pr.shape[3] // 2) - 2]

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_any.append(epe_list)
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        print('Validation FLow360 (%s) EPE: %f' %
              (dstype, epe))
        dstype = 'Flow360' + dstype
        results[dstype] = np.mean(epe_list)

    epe_final_all = np.concatenate(epe_any)
    epe_final = np.mean(epe_final_all)
    print('Validation FLow360 (all) EPE: %f' %
          (epe_final))
