import os
import time
import random
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from util import config
from util.util import (
    AverageMeter,
    intersectionAndUnionGPU,
    save_checkpoint,
    get_palette,
    convert_labels_with_palette_strat,
    extract_clip_feature,
)
from dataset.label_constants import *
from dataset.feature_loader import (
    collate_fn_limit,
    FusedFeatureLoader_strat,
)
from dataset.point_loader import (
    Point3DLoader_strat_withmax_points,
    collation_fn_eval_all,
)
from tqdm import tqdm

import torch_points_kernels as tp
from functools import partial
import wandb
import torch.optim.lr_scheduler as lr_scheduler

best_iou = 0.0


def worker_init_fn(worker_id):
    """Worker initialization."""
    random.seed(time.time() + worker_id)


def get_parser():
    """Parse the config file."""

    parser = argparse.ArgumentParser(description="OpenScene 3D distillation.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/scannet/distill_openseg.yaml",
        help="config file",
    )
    parser.add_argument(
        "opts",
        default=None,
        help="see config/scannet/distill_openseg.yaml for all options",
        nargs=argparse.REMAINDER,
    )
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, "model")
    result_dir = os.path.join(cfg.save_path, "result")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + "/last", exist_ok=True)
    os.makedirs(result_dir + "/best", exist_ok=True)
    return cfg


def get_logger():
    """Define logger."""

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0
    )


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def main():
    """Main function."""

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    # By default we use shared memory for training
    if not hasattr(args, "use_shm"):
        args.use_shm = True

    print(
        "torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s"
        % (
            torch.__version__,
            torch.version.cuda,
            torch.backends.cudnn.version(),
            torch.backends.cudnn.enabled,
        )
    )

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(
            main_worker,
            nprocs=args.ngpus_per_node,
            args=(args.ngpus_per_node, args),
        )
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    args = argss

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    print("RANK", args.rank)
    model = get_model(args)
    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        if args.vis:
            wandb.init(
                project="Openscene",
                name=os.path.basename(args.save_path),
                config=args,
            )
        logger.info(args)
        logger.info("=> creating model ...")

    # ####################### Optimizer ####################### #
    transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "blocks" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "blocks" in n and p.requires_grad
            ],
            "lr": args.base_lr * transformer_lr_scale,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.base_lr, weight_decay=args.weight_decay
    )

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = 1
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu], find_unused_parameters=True
        )
    else:
        model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda()
            )
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_iou = checkpoint["best_iou"]
            if main_process():
                logger.info(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # ####################### Data Loader ####################### #
    if not hasattr(args, "input_color"):
        # by default we do not use the point color as input
        args.input_color = False
    train_data = FusedFeatureLoader_strat(
        datapath_prefix=args.data_root,
        datapath_prefix_feat=args.data_root_2d_fused_feature,
        voxel_size=args.voxel_size,
        voxel_max=args.voxel_max,
        split="train",
        aug=args.aug,
        memcache_init=args.use_shm,
        loop=args.loop,
        input_color=args.input_color,
    )
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_data)
        if args.distributed
        else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=partial(
            collate_fn_limit,
            max_batch_points=args.max_batch_points,
            logger=logger if main_process() else None,
        ),
        worker_init_fn=worker_init_fn,
    )
    if args.evaluate:
        val_data = Point3DLoader_strat_withmax_points(
            datapath_prefix=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max_val,
            split="val",
            aug=False,
            memcache_init=args.use_shm,
            eval_all=True,
            input_color=args.input_color,
        )
        val_sampler = (
            torch.utils.data.distributed.DistributedSampler(val_data)
            if args.distributed
            else None
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collation_fn_eval_all,
            sampler=val_sampler,
        )

        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(
            gpu
        )  # for evaluation

    args.scheduler_update = "epoch"
    milestones = (
        [int(x) for x in args.milestones.split(",")]
        if hasattr(args, "milestones")
        else [int(args.epochs * 0.6), int(args.epochs * 0.8)]
    )
    gamma = args.gamma if hasattr(args, "gamma") else 0.1
    if main_process():
        logger.info(
            "scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(
                args.scheduler_update, milestones, gamma
            )
        )
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    text_features, palette = obtain_text_features_and_palette()
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    # ####################### Distill ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))

        loss_train = distill(
            train_loader,
            model,
            optimizer,
            epoch,
            text_features,
            palette,
            scaler,
            scheduler,
        )

        if args.scheduler_update == "epoch":
            scheduler.step()
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar("loss_train", loss_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader, model, criterion, text_features, palette
            )
            # raise NotImplementedError

            if main_process():
                writer.add_scalar("loss_val", loss_val, epoch_log)
                writer.add_scalar("mIoU_val", mIoU_val, epoch_log)
                writer.add_scalar("mAcc_val", mAcc_val, epoch_log)
                writer.add_scalar("allAcc_val", allAcc_val, epoch_log)
                # remember best iou and save checkpoint
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)
                if is_best:
                    logger.info("Is best")

        if (epoch_log % args.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                },
                is_best,
                os.path.join(args.save_path, "model"),
            )
    if main_process():
        writer.close()
        logger.info("==>Training done!\nBest Iou: %.3f" % (best_iou))


def get_model(args):
    """Get the 3D model."""

    from models.stratified_transformer import Stratified

    args.patch_size = args.grid_size * args.patch_size
    args.window_size = [
        args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)
    ]
    args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
    args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

    if "lseg" in args.feature_2d_extractor:
        last_dim = 512
    elif "openseg" in args.feature_2d_extractor:
        last_dim = 768
    else:
        raise NotImplementedError
    model = Stratified(
        args.downsample_scale,
        args.depths,
        args.channels,
        args.num_heads,
        args.window_size,
        args.up_k,
        args.grid_sizes,
        args.quant_sizes,
        rel_query=args.rel_query,
        rel_key=args.rel_key,
        rel_value=args.rel_value,
        drop_path_rate=args.drop_path_rate,
        concat_xyz=args.concat_xyz,
        num_classes=args.classes,
        ratio=args.ratio,
        k=args.k,
        prev_grid_size=args.grid_size,
        sigma=1.0,
        num_layers=args.num_layers,
        stem_transformer=args.stem_transformer,
        last_dim=last_dim,
    )
    return model


def obtain_text_features_and_palette():
    """obtain the CLIP text feature and palette."""

    if "scannet" in args.data_root:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = "other"
        palette = get_palette()
        dataset_name = "scannet"
    elif "matterport" in args.data_root:
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap="matterport")
        dataset_name = "matterport"
    elif "nuscenes" in args.data_root:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap="nuscenes16")
        dataset_name = "nuscenes"

    if not os.path.exists("saved_text_embeddings"):
        os.makedirs("saved_text_embeddings")

    if "openseg" in args.feature_2d_extractor:
        model_name = "ViT-L/14@336px"
        postfix = "_768"  # the dimension of CLIP features is 768
    elif "lseg" in args.feature_2d_extractor:
        model_name = "ViT-B/32"
        postfix = "_512"  # the dimension of CLIP features is 512
    else:
        raise NotImplementedError

    clip_file_name = "saved_text_embeddings/clip_{}_labels{}.pt".format(
        dataset_name, postfix
    )

    try:  # try to load the pre-saved embedding first
        print("Load pre-computed embeddings from {}".format(clip_file_name))
        text_features = torch.load(clip_file_name).cuda()
    except:  # extract CLIP text features and save them
        text_features = extract_clip_feature(labelset, model_name=model_name)
        torch.save(text_features, clip_file_name)

    return text_features, palette


def distill(
    train_loader,
    model,
    optimizer,
    epoch,
    text_features,
    palette,
    scaler,
    scheduler,
):
    """Distillation pipeline."""

    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    # start the distillation process
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        (coords, feat, label_3d, feat_3d, mask, offset) = batch_data

        # feat from dataset is in [-1,1], change it to [0,1] following COSeg
        feat = (feat + 1) / 2.0
        # coord has additional 1 concated, remove it
        coord = coords[:, 1:].contiguous()

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(
            radius,
            args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[
            0
        ]  # (n, max_num_neighbors)

        coord, feat, label_3d, offset = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            label_3d.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)  # (n, c+3)

        feat_3d, mask = feat_3d.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, output_3d = model(
                feat, coord, offset, batch, neighbor_idx, label_3d
            )  # N_vpt, 512
            output_3d = output_3d[mask]  # N_vvis, 512

            if hasattr(args, "loss_type") and args.loss_type == "cosine":
                loss = (1 - torch.nn.CosineSimilarity()(output_3d, feat_3d)).mean()
            elif hasattr(args, "loss_type") and args.loss_type == "l1":
                loss = torch.nn.L1Loss()(output_3d, feat_3d)
            else:
                raise NotImplementedError

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info(
                "Epoch: [{}/{}][{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Remain {remain_time} "
                "Lr: {lr} "
                "Loss {loss_meter.val:.4f} ".format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    lr=lr,
                    loss_meter=loss_meter,
                )
            )
        if main_process():
            writer.add_scalar("loss_train_batch", loss_meter.val, current_iter)

        end = time.time()

    if args.vis and main_process():
        mask_first = coords[mask][:, 0] == 0
        output_3d = output_3d[mask_first]
        feat_3d = feat_3d[mask_first]
        logits_pred = output_3d.half() @ text_features.t()
        logits_img = feat_3d.half() @ text_features.t()
        logits_pred = torch.max(logits_pred, 1)[1].cpu().numpy()
        logits_img = torch.max(logits_img, 1)[1].cpu().numpy()  # N_vvis
        mask = mask.cpu().numpy()
        logits_gt = label_3d.cpu().numpy()[mask][mask_first.cpu().numpy()]
        logits_gt[logits_gt == 255] = args.classes

        pcl = coords[:, 1:].cpu().numpy()

        # N_vvis,3
        seg_label_color = convert_labels_with_palette_strat(logits_img, palette)
        pred_label_color = convert_labels_with_palette_strat(logits_pred, palette)
        gt_label_color = convert_labels_with_palette_strat(logits_gt, palette)
        pcl_part = pcl[mask][mask_first.cpu().numpy()]

        wandb.log(
            {
                "Feat_2d": wandb.Object3D(
                    np.concatenate([pcl_part, seg_label_color], 1)
                ),
                "Pred": wandb.Object3D(np.concatenate([pcl_part, pred_label_color], 1)),
                "GT": wandb.Object3D(np.concatenate([pcl_part, gt_label_color], 1)),
            }
        )

    return loss_meter.avg


@torch.no_grad()
def validate(val_loader, model, criterion, text_features, palette):
    """Validation."""

    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    model.eval()
    for batch_data in tqdm(val_loader):
        (coords, feat, label, inds_reverse) = batch_data
        # feat from dataset is in [-1,1], change it to [0,1] following COSeg
        feat = (feat + 1) / 2.0
        # coord has additional 1 concated, remove it
        coord = coords[:, 1:].contiguous()

        # only batch size 1, set offset mannually
        offset = torch.tensor([coord.shape[0]], device=feat.device, dtype=torch.int32)

        coord_min = torch.min(coord, 0)[0]
        coord -= coord_min

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(
            radius,
            args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[
            0
        ]  # (n, max_num_neighbors)
        coord, feat, label, offset = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            label.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if args.concat_xyz:
            feat = torch.cat([feat, coord], 1)  # (n, c+3)

        _, output = model(feat, coord, offset, batch, neighbor_idx, label)

        output = output.half() @ text_features.t()
        loss = criterion(output, label)
        output = torch.max(output, 1)[1]

        intersection, union, target = intersectionAndUnionGPU(
            output, label.detach(), args.classes, args.ignore_label
        )
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                target
            )
        intersection, union, target = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            target.cpu().numpy(),
        )
        intersection_meter.update(intersection), union_meter.update(
            union
        ), target_meter.update(target)

        loss_meter.update(loss.item(), args.batch_size_val)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

    if args.vis and main_process():
        pred = output.cpu().numpy()
        gt = label.cpu().numpy()
        gt[gt == 255] = args.classes

        loc = coords[:, 1:].cpu().numpy()

        # N_vvis,3
        pred_label_color = convert_labels_with_palette_strat(pred, palette)
        gt_label_color = convert_labels_with_palette_strat(gt, palette)
        wandb.log(
            {
                "Eval_Pred": wandb.Object3D(np.concatenate([loc, pred_label_color], 1)),
                "Eval_GT": wandb.Object3D(np.concatenate([loc, gt_label_color], 1)),
            }
        )

    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
