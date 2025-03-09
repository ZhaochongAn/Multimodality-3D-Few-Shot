from typing import Optional, Tuple

import numpy as np

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_points_kernels as tp
import wandb

from lib.pointops2.functions import pointops
from model.common import MLPWithoutResidual, KPConvResBlock, MSFLayer
from model.stratified_transformer import Stratified
from util.logger import get_logger


class MM_FSS(nn.Module):
    """
    MultiModal Few-Shot Segmentation (MM-FSS) model.
    """

    def __init__(self, args):
        super(MM_FSS, self).__init__()
        self.args = args

        # Few-shot segmentation parameters
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_subprototypes = args.n_subprototypes
        self.n_queries = args.n_queries
        self.n_classes = self.n_way + 1

        # Loss functions
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1] + [1 for _ in range(self.n_way)]),
            ignore_index=args.ignore_label,
        )
        self.criterion_base = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        # Adjust arguments for patch and window sizes
        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)
        ]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        # Set up base class mappings for different datasets
        if args.data_name == "s3dis":
            self.base_classes = 6
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    0: 1,
                    3: 2,
                    4: 3,
                    8: 4,
                    10: 5,
                    11: 6,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1,
                    2: 2,
                    5: 3,
                    6: 4,
                    7: 5,
                    9: 6,
                }
        else:
            self.base_classes = 10
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    2: 1,
                    3: 2,
                    5: 3,
                    6: 4,
                    7: 5,
                    10: 6,
                    12: 7,
                    13: 8,
                    14: 9,
                    19: 10,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1,
                    4: 2,
                    8: 3,
                    9: 4,
                    11: 5,
                    15: 6,
                    16: 7,
                    17: 8,
                    18: 9,
                    20: 10,
                }

        # Set logger if in main process
        if self.main_process():
            self.logger = get_logger(args.save_path)

        # Text model dimensionality
        if args.textmodel == "openseg":
            last_dim = 768
        else:
            last_dim = 512

        # 3D backbone (Stratified Transformer)
        self.encoder = Stratified(
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
            num_classes=self.base_classes + 1,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
            logger=get_logger(args.save_path),
            last_dim=last_dim,
        )

        self.feat_dim = args.channels[2]
        self.visualization = args.vis

        self.lin1 = nn.Sequential(
            nn.Linear(self.n_subprototypes, self.feat_dim),
            nn.ReLU(inplace=True),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(self.n_subprototypes, self.feat_dim),
            nn.ReLU(inplace=True),
        )

        # KPConv block for feature refinement
        self.kpconv = KPConvResBlock(self.feat_dim, self.feat_dim, 0.04, sigma=2)

        # Final classification head
        self.cls = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.feat_dim, self.n_classes),
        )

        # A small FFN for the backbone features
        self.bk_ffn = nn.Sequential(
            nn.Linear(self.feat_dim + (self.feat_dim // 2), 4 * self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * self.feat_dim, self.feat_dim),
        )

        # Number of MSF layers differs by dataset
        if self.args.data_name == "s3dis":
            agglayers = 2
        else:
            agglayers = 4
        print(f"Using {agglayers} MSF layers for aggregator fusion.")

        # Build the MSF layers
        self.msf_layers = nn.ModuleList(
            [
                MSFLayer(
                    hidden_dim=self.feat_dim,
                    nheads=4,
                    attention_type="linear",
                    num_classes=self.n_classes,
                )
                for _ in range(agglayers)
            ]
        )

        # Class reducer
        if self.n_way == 1:
            self.class_reduce = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Conv1d(self.n_classes, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.class_reduce = MLPWithoutResidual(
                self.feat_dim * (self.n_way + 1), self.feat_dim
            )

        # MLPs for background prototype reduction
        self.bg_proto_reduce = MLPWithoutResidual(
            self.n_subprototypes * self.n_way, self.n_subprototypes
        )
        self.IF_bg_proto_reduce = MLPWithoutResidual(
            self.n_subprototypes * self.n_way, self.n_subprototypes
        )

        self.init_weights()

        self.register_buffer(
            "base_prototypes", torch.zeros(self.base_classes, self.feat_dim)
        )

        # Load text-based features
        self.background = self.extract_clip_feature(["other"], args.textmodel)
        self.label_feats = self.precompute_text_related_properties(
            args.data_name, args.textmodel
        )

    def init_weights(self):
        """
        Initialize learnable parameters in the network.
        """
        for name, m in self.named_parameters():
            if "transformer_layer.base_merge" in name:
                continue
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def main_process(self):
        return not self.args.multiprocessing_distributed or (
            self.args.multiprocessing_distributed
            and self.args.rank % self.args.ngpus_per_node == 0
        )

    def precompute_text_related_properties(self, labelset_name, feature_2d_extractor):
        """
        Precompute text features for all class labels in the dataset.

        Args:
            labelset_name: Name of the label set (e.g., 's3dis' or 'scannetv2').
            feature_2d_extractor: Which 2D VLM' s text encoder to use (e.g., 'openseg', 'lseg').

        Returns:
            text_features: A tensor of shape (num_classes, clip_dim) containing text embeddings.
        """
        if labelset_name == "scannetv2":
            from model.label_constants import (
                SCANNET_LABELS_20_coseg as SCANNET_LABELS_20,
            )

            labelset = list(SCANNET_LABELS_20)
            labelset[-1] = "other"  # rename 'other furniture' to 'other'
        elif labelset_name == "s3dis":
            from model.label_constants import S3DIS_LABELS_12

            labelset = list(S3DIS_LABELS_12)
        else:
            raise NotImplementedError("Unsupported dataset name.")

        # Simple prompt engineering
        print("Using prompt engineering: 'a XX in a scene'")
        labelset = ["a " + lb + " in a scene" for lb in labelset]
        if "other" in labelset[-1]:
            labelset[-1] = "other"

        text_features = self.extract_clip_feature(labelset, feature_2d_extractor)
        return text_features

    @torch.no_grad()
    def extract_clip_feature(self, labelset, feature_2d_extractor):
        """
        Extract text features from a text encoder.

        Args:
            labelset: List of strings or a single string.
            feature_2d_extractor: Which pretrained 2D VLM to use (e.g., 'openseg', 'lseg').

        Returns:
            text_features: L2-normalized text embeddings for the given labels.
        """
        if "openseg" in feature_2d_extractor:
            model_name = "ViT-L/14@336px"
        elif "lseg" in feature_2d_extractor:
            model_name = "ViT-B/32"  # or change to your pre-downloaded model path
        else:
            raise NotImplementedError(
                f"Unknown feature_2d_extractor: {feature_2d_extractor}"
            )

        print(f"Loading CLIP model from {model_name}...")
        clip_pretrained, _ = clip.load(model_name, device="cuda", jit=False)
        print("Finished loading CLIP model.")

        if isinstance(labelset, str):
            lines = labelset.split(",")
        elif isinstance(labelset, list):
            lines = labelset
        else:
            raise NotImplementedError("Labelset must be a string or list of strings.")

        # Tokenize labels
        text = clip.tokenize(lines).cuda()
        text_features = clip_pretrained.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def forward(
        self,
        support_offset: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_offset: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        epoch: int,
        support_base_y: Optional[torch.Tensor] = None,
        query_base_y: Optional[torch.Tensor] = None,
        sampled_classes: Optional[np.array] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MM_FSS model.

        Args:
            support_offset: Offset of each scene in the support set (shape: [N_way*K_shot]).
            support_x: Support point cloud inputs (shape: [N_support, in_channels]).
            support_y: Support masks (shape: [N_support]).
            query_offset: Offset of each scene in the query set (shape: [N_way]).
            query_x: Query point cloud inputs (shape: [N_query, in_channels]).
            query_y: Query labels (shape: [N_query]).
            epoch: Current training epoch.
            support_base_y: Base class labels in the support set (shape: [N_support]).
            query_base_y: Base class labels in the query set (shape: [N_query]).
            sampled_classes: The classes sampled in the current episode (shape: [N_way]).

        Returns:
            final_pred: Predicted class logits for query point clouds (shape: [1, n_way+1, N_query]).
            loss: The total loss value for this forward pass.
        """
        # ---------------------------------------------------------------------
        # 1. Get support features via backbone encoder
        # ---------------------------------------------------------------------
        (
            support_feat,  # N_s, C
            support_x_low,  # N_s, 3
            support_offset_low,
            support_y_low,  # N_s
            _,
            support_base_y,  # N_s
            support_IF_feats,  # N_q, C
        ) = self.getFeatures(support_x, support_offset, support_y, support_base_y)

        # Split support features into scenes
        support_offset_low = support_offset_low[:-1].long().cpu()
        support_feat = torch.tensor_split(support_feat, support_offset_low)
        support_x_low = torch.tensor_split(support_x_low, support_offset_low)
        support_IF_feats = torch.tensor_split(support_IF_feats, support_offset_low)
        if support_base_y is not None:
            support_base_y = torch.tensor_split(support_base_y, support_offset_low)

        # Create foreground/background masks
        fg_mask = support_y_low
        bg_mask = torch.logical_not(support_y_low)
        fg_mask = torch.tensor_split(fg_mask, support_offset_low)
        bg_mask = torch.tensor_split(bg_mask, support_offset_low)

        # ---------------------------------------------------------------------
        # 2. Construct prototypes from the support set (For k_shot, extract N_pt/k_shot per shot)
        # ---------------------------------------------------------------------
        fg_prototypes, IF_fg_prototypes = self.getPrototypes(
            support_x_low,
            support_feat,
            support_IF_feats,
            fg_mask,
            k=self.n_subprototypes // self.k_shot,
        )  # N_way*N_pt, C
        bg_prototype, IF_bg_prototype = self.getPrototypes(
            support_x_low,
            support_feat,
            support_IF_feats,
            bg_mask,
            k=self.n_subprototypes // self.k_shot,
        )  # N_way*N_pt, C

        # If multiple ways, reduce the number of BG prototypes to n_subprototypes
        if bg_prototype.shape[0] > self.n_subprototypes:
            bg_prototype = self.bg_proto_reduce(bg_prototype.permute(1, 0)).permute(
                1, 0
            )
            IF_bg_prototype = self.IF_bg_proto_reduce(
                IF_bg_prototype.permute(1, 0)
            ).permute(1, 0)

        # Combine BG + FG prototypes
        sparse_embeddings = torch.cat(
            [bg_prototype, fg_prototypes]
        )  # (N_way+1)*N_pt, C
        IF_sparse_embeddings = torch.cat(
            [IF_bg_prototype, IF_fg_prototypes]
        )  # (N_way+1)*N_pt, C

        # ---------------------------------------------------------------------
        # 3. Get query features via backbone encoder
        # ---------------------------------------------------------------------
        (
            query_feat,  # N_q, C
            query_x_low,  # N_q, 3
            query_offset_low,
            query_y_low,  # N_q
            q_base_pred,  # N_q, N_base_classes
            query_base_y,  # N_q
            query_IF_feats,  # N_q, C
        ) = self.getFeatures(query_x, query_offset, query_y, query_base_y)

        # Split query features into scenes
        query_offset_low_cpu = query_offset_low[:-1].long().cpu()
        query_feat = torch.tensor_split(query_feat, query_offset_low_cpu)
        query_x_low_list = torch.tensor_split(query_x_low, query_offset_low_cpu)
        query_IF_feats = torch.tensor_split(query_IF_feats, query_offset_low_cpu)
        if query_base_y is not None:
            query_base_y_list = torch.tensor_split(query_base_y, query_offset_low_cpu)

        # ---------------------------------------------------------------------
        # 4. Update base prototypes (during training)
        # ---------------------------------------------------------------------
        if self.training:
            for base_feat, base_y in zip(
                list(query_feat) + list(support_feat),
                list(query_base_y_list) + list(support_base_y),
            ):
                cur_baseclsses = base_y.unique()
                cur_baseclsses = cur_baseclsses[
                    cur_baseclsses != 0
                ]  # remove background
                for class_label in cur_baseclsses:
                    class_mask = base_y == class_label
                    class_features = (
                        base_feat[class_mask].sum(dim=0) / class_mask.sum()
                    ).detach()  # C
                    # Initialize or update base_prototypes with EMA
                    if torch.all(self.base_prototypes[class_label - 1] == 0):
                        self.base_prototypes[class_label - 1] = class_features
                    else:
                        self.base_prototypes[class_label - 1] = (
                            self.base_prototypes[class_label - 1] * 0.995
                            + class_features * 0.005
                        )
            # Mask out base prototypes for current target classes which should not be considered as background
            mask_list = [
                self.base_class_to_pred_label[base_cls] - 1
                for base_cls in sampled_classes
            ]
            base_mask = self.base_prototypes.new_ones(
                (self.base_prototypes.shape[0]), dtype=torch.bool
            )
            base_mask[mask_list] = False
            base_avail_pts = self.base_prototypes[base_mask]
            assert len(base_avail_pts) == self.base_classes - self.n_way
        else:
            base_avail_pts = self.base_prototypes

        # Choose text embeddings for the sampled classes
        if self.args.data_name == "s3dis":
            clip_label_feats = self.label_feats[sampled_classes]
        else:
            # For ScanNet classes start from 1
            clip_label_feats = self.label_feats[sampled_classes - 1]

        # ---------------------------------------------------------------------
        # 5. Perform few-shot classification on each query scene
        # ---------------------------------------------------------------------
        device = self.background.device
        dtype = self.label_feats.dtype

        query_guidance = []
        query_pred = []

        for i, q_feat in enumerate(query_feat):
            # Intermodal query features -> correlation with BG + FG text embeddings -> semantic guidance
            query_guidance.append(
                query_IF_feats[i].to(dtype)
                @ torch.cat([self.background, clip_label_feats]).T
            )  # N_q, N_way+1

            # Base guidance (cosine similarity to base prototypes),
            # only used after a warm-up epoch
            if epoch < self.args.base_guidance_start_epoch:
                base_guidance = None
            else:
                base_similarity = F.cosine_similarity(
                    q_feat[:, None, :],  # N_q, 1, C
                    base_avail_pts[None, :, :],  # 1, N_base_classes, C
                    dim=2,
                )  # N_q, N_base_classes
                # max similarity for each query point as the base guidance
                base_guidance = base_similarity.max(dim=1, keepdim=True)[0]  # N_q, 1

            # Multimodal correlations
            UF_correlations = F.cosine_similarity(
                q_feat[:, None, :],
                sparse_embeddings[None, :, :],  # 1, (N_way+1)*N_pt, C
                dim=2,
            )  # N_q, (N_way+1)*N_pt
            IF_correlations = F.cosine_similarity(
                query_IF_feats[i][:, None, :],
                IF_sparse_embeddings[None, :, :],  # 1, (N_way+1)*N_pt, C
                dim=2,
            )  # N_q, (N_way+1)*N_pt

            UF_correlations = (
                self.lin1(
                    UF_correlations.view(
                        UF_correlations.shape[0], self.n_way + 1, -1
                    )  # N_q, (N_way+1), N_pt
                )
                .permute(2, 1, 0)
                .unsqueeze(0)
            )  # 1, C, N_way+1, N_q

            IF_correlations = (
                self.lin2(
                    IF_correlations.view(
                        IF_correlations.shape[0], self.n_way + 1, -1
                    )  # N_q, (N_way+1), N_pt
                )
                .permute(2, 1, 0)
                .unsqueeze(0)
            )  # 1, C, N_way+1, N_q

            # Get multimodal correlations
            correlations = UF_correlations + IF_correlations

            # MSF layers
            for layer in self.msf_layers:
                correlations = layer(
                    correlations,
                    base_guidance,
                    query_guidance[i],
                )  # 1, C, N_way+1, N_q

            correlations = (
                correlations.squeeze(0).permute(2, 1, 0).contiguous()
            )  # N_q, N_way+1, C

            # Reduce class dimension
            if self.n_way == 1:
                correlations = self.class_reduce(correlations).squeeze(1)  # N_q, C
            else:
                correlations = self.class_reduce(
                    correlations.view(correlations.shape[0], -1)
                )  # N_q, C

            # KPConv layer
            coord = query_x_low_list[i]  # N_q, 3
            batch = torch.zeros(
                correlations.shape[0], dtype=torch.int64, device=coord.device
            )
            sigma = 2.0
            radius = 2.5 * self.args.grid_size * sigma
            neighbors = tp.ball_query(
                radius,
                self.args.max_num_neighbors,
                coord,
                coord,
                mode="partial_dense",
                batch_x=batch,
                batch_y=batch,
            )[
                0
            ]  # N_q, max_num_neighbors
            correlations = self.kpconv(
                correlations, coord, batch, neighbors.clone()
            )  # N_q, C

            # Final classifier
            out = self.cls(correlations)  # N_q, n_way+1
            query_pred.append(out)

        # ---------------------------------------------------------------------
        # 6. Compute losses
        # ---------------------------------------------------------------------
        query_pred = torch.cat(query_pred)  # N_q, n_way+1
        loss = self.criterion(query_pred, query_y_low)
        if query_base_y is not None:
            loss += self.criterion_base(q_base_pred, query_base_y.cuda())

        final_pred = (
            pointops.interpolation(
                query_x_low,
                query_x[:, :3].cuda().contiguous(),
                query_pred.contiguous(),
                query_offset_low,
                query_offset.cuda(),
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )  # 1, n_way+1, N_query

        if self.training:
            return final_pred, loss

        # ---------------------------------------------------------------------
        # 7. Test-time Adaptive Cross-modal Calibration
        # ---------------------------------------------------------------------
        support_y_low = torch.tensor_split(support_y_low, support_offset_low)
        iou_list = []
        for nw in range(self.n_way):
            cur_way_ious = []
            if self.args.data_name == "s3dis":
                cur_clip_feat = self.label_feats[sampled_classes[nw]][None]
            else:
                cur_clip_feat = self.label_feats[sampled_classes[nw] - 1][
                    None
                ]  # Scannet clasees index start from 1

            for k_shot_idx in range(self.k_shot):
                support_guidance = (
                    support_IF_feats[nw * self.k_shot + k_shot_idx].to(dtype)
                    @ torch.cat([self.background, cur_clip_feat]).T
                )
                support_guidance = torch.max(support_guidance, 1)[1]
                shot_label = support_y_low[nw * self.k_shot + k_shot_idx].to(device)
                iou = sum(support_guidance & shot_label) / sum(
                    support_guidance | shot_label
                )
                cur_way_ious.append(iou)
            iou_list.append(max(cur_way_ious))

        query_guidance = torch.cat(query_guidance)
        query_guidance = pointops.interpolation(
            query_x_low,
            query_x[:, :3].cuda().contiguous(),
            query_guidance.contiguous(),
            query_offset_low,
            query_offset.cuda(),
        )

        final_pred = final_pred.squeeze(0).T  # N_q, n_way+1

        final_pred = torch.tensor_split(
            final_pred, query_offset[:-1].long().cpu(), dim=0
        )
        query_guidance = torch.tensor_split(
            query_guidance, query_offset[:-1].long().cpu(), dim=0
        )

        # Combine final_pred and query_guidance
        calibrated_pred = []
        for i in range(self.n_way):
            calibrated_pred.append(final_pred[i] + iou_list[i] * query_guidance[i])

        final_pred = torch.cat(calibrated_pred).T.unsqueeze(0)  # 1, n_way+1, N_q

        # wandb visualization
        if self.visualization:
            self.vis(
                query_offset,
                query_x,
                query_y,
                support_offset,
                support_x,
                support_y,
                final_pred,
            )

        return final_pred, loss

    def getFeatures(self, ptclouds, offset, gt, query_base_y=None):
        """
        Get the features of one point cloud from backbone network.

        Args:
            ptclouds: Input point clouds with shape (N_pt, 6), where N_pt is the number of points.
            offset: Offset tensor with shape (b), where b is the number of query scenes.
            gt: Ground truth labels. shape (N_pt).
            query_base_y: Optional base class labels for input point cloud. shape (N_pt).

        Returns:
            feat: Features from backbone with shape (N_down, C), where C is the number of channels.
            coord: Point coords. Shape (N_down, 3).
            offset: Offset for each scene. Shape (b).
            gt: Ground truth labels. Shape (N_down).
            base_pred: Base class predictions from backbone. Shape (N_down, N_base_classes).
            query_base_y: Base class labels for input point cloud. Shape (N_down).
            IF_feats: Features from IF head. Shape (N_down, C).
        """
        coord, feat = (
            ptclouds[:, :3].contiguous(),
            ptclouds[:, 3:6].contiguous(),  # rgb color
        )  # (N_pt, 3), (N_pt, 3)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()  # N_pt

        sigma = 1.0
        radius = 2.5 * self.args.grid_size * sigma
        batch = batch.to(coord.device)
        neighbor_idx = tp.ball_query(
            radius,
            self.args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[
            0
        ]  # (N_pt, max_num_neighbors)

        coord, feat, offset, gt = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
            gt.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if self.args.concat_xyz:
            feat = torch.cat([feat, coord], 1)  # N_pt, 6
        # downsample the input point clouds
        (
            feat_cat,
            coord,
            offset,
            gt,
            base_pred,
            query_base_y,
            IF_feats,
        ) = self.encoder(
            feat, coord, offset, batch, neighbor_idx, gt, query_base_y
        )  # (N_down, C_bc) (N_down, 3) (b), (N_down), (N_down, N_base_classes), (N_down)
        feat_cat = self.bk_ffn(feat_cat)  # N_down, C

        return (
            feat_cat,
            coord,
            offset,
            gt,
            base_pred,
            query_base_y,
            IF_feats,
        )

    def getPrototypes(self, coords, feats, IF_feats, masks, k=100):
        """
        Extract k prototypes for each scene.

        Args:
            coords: Point coordinates. List of (N_pt, 3).
            feats: Unimodal point features. List of (N_pt, C).
            IF_feats: Intermodal point features. List of (N_pt, C).
            masks: Target class masks. List of (N_pt).
            k: Number of prototypes extracted in each shot (default: 100).

        Return:
            prototypes: Shape (n_way*k_shot*k, C).
            IF_prototypes: Shape (n_way*k_shot*k, C).
        """
        prototypes = []
        IF_prototypes = []
        for i in range(0, self.n_way * self.k_shot):
            coord = coords[i][:, :3]  # N_pt, 3
            feat = feats[i]  # N_pt, C
            mask = masks[i].bool()  # N_pt
            IF_feat = IF_feats[i]

            coord_mask = coord[mask]
            feat_mask = feat[mask]
            IF_feat_mask = IF_feat[mask]
            protos, IF_protos = self.getMutiplePrototypes(
                coord_mask, feat_mask, IF_feat_mask, k
            )  # k, C
            prototypes.append(protos)
            IF_prototypes.append(IF_protos)

        prototypes = torch.cat(prototypes)  # n_way*k_shot*k, C
        IF_prototypes = torch.cat(IF_prototypes)
        return prototypes, IF_prototypes

    def getMutiplePrototypes(self, coord, feat, IF_feat, num_prototypes):
        """
        Extract k prototypes using furthest point samplling

        Args:
            coord: Point coordinates. Shape (N_pt, 3)
            feat: Unimodal point features. Shape (N_pt, C).
            IF_feat: Intermodal point features. Shape (N_pt, C).
            num_prototypes: Number of prototypes to extract.

        Return:
            prototypes: Extracted unimodal prototypes. Shape: (num_prototypes, C).
            IF_prototypes: Extracted intermodal prototypes. Shape: (num_prototypes, C).
        """
        # when the number of points is less than the number of prototypes, pad the points with zero features
        if feat.shape[0] <= num_prototypes:
            no_feats = feat.new_zeros(
                1,
                self.feat_dim,
            ).expand(num_prototypes - feat.shape[0], -1)
            feat = torch.cat([feat, no_feats])

            no_IF_feats = IF_feat.new_zeros(
                1,
                IF_feat.shape[1],
            ).expand(num_prototypes - IF_feat.shape[0], -1)
            IF_feat = torch.cat([IF_feat, no_IF_feats])
            return feat, IF_feat

        # sample k seeds by Farthest Point Sampling
        fps_index = pointops.furthestsampling(
            coord,
            torch.cuda.IntTensor([coord.shape[0]]),
            torch.cuda.IntTensor([num_prototypes]),
        ).long()  # (num_prototypes,)

        # use the k seeds as initial centers and compute the point-to-seed distance
        num_prototypes = len(fps_index)

        farthest_seeds = feat[fps_index]  # (num_prototypes, feat_dim)
        distances = torch.linalg.norm(
            feat[:, None, :] - farthest_seeds[None, :, :], dim=2
        )  # (N_pt, num_prototypes)

        # clustering the points to the nearest seed
        assignments = torch.argmin(distances, dim=1)  # (N_pt,)

        # aggregating each cluster to form prototype
        prototypes = torch.zeros((num_prototypes, self.feat_dim), device="cuda")
        for i in range(num_prototypes):
            selected = torch.nonzero(assignments == i).squeeze(1)  # (N_selected,)
            selected = feat[selected, :]  # (N_selected, C)
            if (
                len(selected) == 0
            ):  # exists same prototypes (coord not same), points are assigned to the prior prototype
                # simple use the seed as the prototype here
                prototypes[i] = feat[fps_index[i]]
                if self.main_process():
                    self.logger.info("len(selected) == 0")
            else:
                prototypes[i] = selected.mean(0)  # (C,)

        # IF prototypes generation
        farthest_seeds = IF_feat[fps_index]  # (num_prototypes, feat_dim)
        distances = torch.linalg.norm(
            IF_feat[:, None, :] - farthest_seeds[None, :, :], dim=2
        )  # (N_pt, num_prototypes)

        # clustering the points to the nearest seed
        assignments = torch.argmin(distances, dim=1)  # (N_pt,)

        # aggregating each cluster to form prototype
        IF_prototypes = torch.zeros((num_prototypes, IF_feat.shape[1]), device="cuda")
        for i in range(num_prototypes):
            selected = torch.nonzero(assignments == i).squeeze(1)  # (N_selected,)
            selected = IF_feat[selected, :]  # (N_selected, C)
            if (
                len(selected) == 0
            ):  # exists same prototypes (coord not same), points are assigned to the prior prototype
                # simple use the seed as the prototype here
                IF_prototypes[i] = IF_feat[fps_index[i]]
                if self.main_process():
                    self.logger.info("len(selected) == 0")
            else:
                IF_prototypes[i] = selected.mean(0)  # (C,)
        return prototypes, IF_prototypes

    def vis(
        self,
        query_offset,
        query_x,
        query_y,
        support_offset,
        support_x,
        support_y,
        final_pred,
    ):
        query_offset_cpu = query_offset[:-1].long().cpu()
        query_x_splits = torch.tensor_split(query_x, query_offset_cpu)
        query_y_splits = torch.tensor_split(query_y, query_offset_cpu)
        vis_pred = torch.tensor_split(final_pred, query_offset_cpu, dim=-1)
        support_offset_cpu = support_offset[:-1].long().cpu()
        vis_mask = torch.tensor_split(support_y, support_offset_cpu)

        sp_nps, sp_fgs = [], []
        for i, support_x_split in enumerate(
            torch.tensor_split(support_x, support_offset_cpu)
        ):
            sp_np = support_x_split.detach().cpu().numpy()  # num_points, in_channels
            sp_np[:, 3:6] = sp_np[:, 3:6] * 255.0
            sp_fg = np.concatenate(
                (
                    sp_np[:, :3],
                    vis_mask[i].unsqueeze(-1).detach().cpu().numpy(),
                ),
                axis=-1,
            )
            sp_nps.append(sp_np)
            sp_fgs.append(sp_fg)

        qu_s, qu_gts, qu_pds = [], [], []
        for i, query_x_split in enumerate(query_x_splits):
            qu = query_x_split.detach().cpu().numpy()  # num_points, in_channels
            qu[:, 3:6] = qu[:, 3:6] * 255.0
            result_tensor = torch.where(
                query_y_splits[i] == 255,
                torch.tensor(0, device=query_y.device),
                query_y_splits[i],
            )
            qu_gt = np.concatenate(
                (
                    qu[:, :3],
                    result_tensor.unsqueeze(-1).detach().cpu().numpy(),
                ),
                axis=-1,
            )
            q_prd = np.concatenate(
                (
                    qu[:, :3],
                    vis_pred[i]
                    .squeeze(0)
                    .max(0)[1]
                    .unsqueeze(-1)
                    .detach()
                    .cpu()
                    .numpy(),
                ),
                axis=-1,
            )

            qu_s.append(qu)
            qu_gts.append(qu_gt)
            qu_pds.append(q_prd)

        wandb.log(
            {
                "Support": [wandb.Object3D(sp_nps[i]) for i in range(len(sp_nps))],
                "Support_fg": [wandb.Object3D(sp_fgs[i]) for i in range(len(sp_fgs))],
                "Query": [wandb.Object3D(qu_s[i]) for i in range(len(qu_s))],
                "Query_pred": [wandb.Object3D(qu_pds[i]) for i in range(len(qu_pds))],
                "Query_GT": [wandb.Object3D(qu_gts[i]) for i in range(len(qu_gts))],
            }
        )
