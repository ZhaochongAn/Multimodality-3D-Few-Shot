import os
import copy
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.init as initer
import torch.nn.functional as F


def evaluate_metric(pred_label, gt_label, label2class, test_classes, ignore_index=255):
    """_summary_

    :param pred_label: shape (n_queries*n_way * num_points).
    :param gt_label: shape (n_queries*n_way * num_points).
    :param label2class: sampled original class id in one few-shot episode
    :param test_classes: the original id of all test classes
    :return: _description_
    """

    NUM_CLASS = len(test_classes)
    test_classes = {c: i for i, c in enumerate(test_classes)}

    union_classes = torch.tensor([0 for _ in range(NUM_CLASS)], device="cuda")
    intersec_classes = torch.tensor([0 for _ in range(NUM_CLASS)], device="cuda")
    target_classes = torch.tensor([0 for _ in range(NUM_CLASS)], device="cuda")
    intersection, union, target = intersectionAndUnionGPU(
        pred_label, gt_label, len(label2class) + 1, ignore_index
    )
    for i in range(1, len(intersection)):
        class_idx = test_classes[label2class[i - 1]]
        intersec_classes[class_idx] += intersection[i]
        union_classes[class_idx] += union[i]
        target_classes[class_idx] += target[i]

    return intersec_classes, union_classes, target_classes


def load_pretrain_checkpoint(model, pretrain_checkpoint_path, gpu):
    # load pretrained model for point cloud encoding
    model_dict = model.state_dict()
    if pretrain_checkpoint_path is not None:
        print("Load encoder module from pretrained checkpoint...")
        if isinstance(gpu, list):
            assert len(gpu) == 1
            gpu = gpu[0]
        pretrained_dict = torch.load(
            os.path.join(pretrain_checkpoint_path, "model", "model_best.pth.tar"),
            map_location={"cuda:0": "cuda:%d" % gpu},
        )["state_dict"]

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            pretrained_dict = {
                k.replace("module.", "module.encoder."): v
                for k, v in pretrained_dict.items()
            }
        else:
            pretrained_dict = {
                k.replace("module.", "encoder."): v for k, v in pretrained_dict.items()
            }

        # remove key value pair if classifier in key
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if "classifier" not in k
        }

        assert (
            set(pretrained_dict.keys()).issubset(set(model_dict.keys()))
            and len(pretrained_dict.keys()) > 40
        )
        model_dict.update(pretrained_dict)

        UF_head_keys = [key for key in model_dict.keys() if "UF_head" in key]

        # Initialize weights for UF_head by copying from IF_head
        for UF_head_key in UF_head_keys:
            orginal_key = UF_head_key.replace("UF_head", "layers.2")
            model_dict[UF_head_key] = copy.deepcopy(model_dict[orginal_key])

        model.load_state_dict(model_dict)

    else:
        raise ValueError("Pretrained checkpoint must be given.")

    return model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(
    optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6
):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    # import pdb; pdb.set_trace()
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(
    model, conv="kaiming", batchnorm="normal", linear="kaiming", lstm="kaiming"
):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (_ConvNd)):
            if conv == "kaiming":
                initer.kaiming_normal_(m.weight)
            elif conv == "xavier":
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, _BatchNorm):
            if batchnorm == "normal":
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == "constant":
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == "kaiming":
                initer.kaiming_normal_(m.weight)
            elif linear == "xavier":
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight" in name:
                    if lstm == "kaiming":
                        initer.kaiming_normal_(param)
                    elif lstm == "xavier":
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif "bias" in name:
                    initer.constant_(param, 0)


def convert_to_syncbn(model):
    def recursive_set(cur_module, name, module):
        if len(name.split(".")) > 1:
            recursive_set(
                getattr(cur_module, name[: name.find(".")]),
                name[name.find(".") + 1 :],
                module,
            )
        else:
            setattr(cur_module, name, module)

    from lib.sync_bn import (
        SynchronizedBatchNorm1d,
        SynchronizedBatchNorm2d,
        SynchronizedBatchNorm3d,
    )

    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            recursive_set(
                model,
                name,
                SynchronizedBatchNorm1d(m.num_features, m.eps, m.momentum, m.affine),
            )
        elif isinstance(m, nn.BatchNorm2d):
            recursive_set(
                model,
                name,
                SynchronizedBatchNorm2d(m.num_features, m.eps, m.momentum, m.affine),
            )
        elif isinstance(m, nn.BatchNorm3d):
            recursive_set(
                model,
                name,
                SynchronizedBatchNorm3d(m.num_features, m.eps, m.momentum, m.affine),
            )


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert("P")
    color.putpalette(palette)
    return color


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def memory_use():
    BYTES_IN_GB = 1024**3
    return "ALLOCATED: {:>6.3f} ({:>6.3f})  CACHED: {:>6.3f} ({:>6.3f})".format(
        torch.cuda.memory_allocated() / BYTES_IN_GB,
        torch.cuda.max_memory_allocated() / BYTES_IN_GB,
        torch.cuda.memory_reserved() / BYTES_IN_GB,
        torch.cuda.max_memory_reserved() / BYTES_IN_GB,
    )


def smooth_loss(output, target, eps=0.1):
    w = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
    w = w * (1 - eps) + (1 - w) * eps / (output.shape[1] - 1)
    log_prob = F.log_softmax(output, dim=1)
    loss = (-w * log_prob).sum(dim=1).mean()
    return loss
