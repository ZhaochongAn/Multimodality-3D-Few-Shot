"""Dataloader for fused point features."""

import copy
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA
import wandb

from dataset.point_loader import Point3DLoader
from dataset.voxelize import feat_point_voxelize


class FusedFeatureLoader(Point3DLoader):
    """Dataloader for fused point features."""

    def __init__(
        self,
        datapath_prefix,
        datapath_prefix_feat,
        voxel_size=0.05,
        split="train",
        aug=False,
        memcache_init=False,
        identifier=7791,
        loop=1,
        eval_all=False,
        input_color=False,
    ):
        super().__init__(
            datapath_prefix=datapath_prefix,
            voxel_size=voxel_size,
            split=split,
            aug=aug,
            memcache_init=memcache_init,
            identifier=identifier,
            loop=loop,
            eval_all=eval_all,
            input_color=input_color,
        )
        self.aug = aug
        self.input_color = (
            input_color  # decide whether we use point color values as input
        )

        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        # Precompute the occurances for each scene
        # for training sets, ScanNet and Matterport has 5 each, nuscene 1
        # for evaluation/test sets, all has just one
        if "nuscenes" in self.dataset_name:  # only one file for each scene
            self.list_occur = None
        else:
            self.list_occur = []
            for data_path in self.data_paths:
                if "scannet" in self.dataset_name:
                    scene_name = data_path[:-15].split("/")[-1]
                else:
                    scene_name = data_path[:-4].split("/")[-1]
                file_dirs = glob(join(self.datapath_feat, scene_name + "_*.pt"))
                self.list_occur.append(len(file_dirs))
            # some scenes in matterport have no features at all
            ind = np.where(np.array(self.list_occur) != 0)[0]
            if np.any(np.array(self.list_occur) == 0):
                data_paths, list_occur = [], []
                for i in ind:
                    data_paths.append(self.data_paths[i])
                    list_occur.append(self.list_occur[i])
                self.data_paths = data_paths
                self.list_occur = list_occur

        if len(self.data_paths) == 0:
            raise Exception("0 file is loaded in the feature loader.")

    def __getitem__(self, index_long):
        index = index_long % len(self.data_paths)
        if self.use_shm:
            locs_in = SA.attach(
                "shm://%s_%s_%06d_locs_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
            feats_in = SA.attach(
                "shm://%s_%s_%06d_feats_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
            labels_in = SA.attach(
                "shm://%s_%s_%06d_labels_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
        else:
            # (N_pt, 3), (N_pt, 3), (N_pt,)
            locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            if np.isscalar(feats_in) and feats_in == 0:
                # no color in the input point cloud, e.g nuscenes lidar
                feats_in = np.zeros_like(locs_in)
            else:
                feats_in = (feats_in + 1.0) * 127.5

        # load 3D features
        if self.dataset_name == "scannet_3d":
            scene_name = self.data_paths[index][:-15].split("/")[-1]
        else:
            scene_name = self.data_paths[index][:-4].split("/")[-1]

        if "nuscenes" not in self.dataset_name:
            n_occur = self.list_occur[index]
            if n_occur > 1:
                nn_occur = np.random.randint(n_occur)
            elif n_occur == 1:
                nn_occur = 0
            else:
                raise NotImplementedError

            processed_data = torch.load(
                join(self.datapath_feat, scene_name + "_%d.pt" % (nn_occur))
            )
        else:
            # no repeated file
            processed_data = torch.load(join(self.datapath_feat, scene_name + ".pt"))

        flag_mask_merge = False
        if len(processed_data.keys()) == 2:
            flag_mask_merge = True
            feat_3d, mask_chunk = (
                processed_data["feat"],
                processed_data["mask_full"],
            )
            if isinstance(
                mask_chunk, np.ndarray
            ):  # if the mask itself is a numpy array
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
            if self.split != "train":  # val or test set
                feat_3d_new = torch.zeros(
                    (locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype
                )
                feat_3d_new[mask] = feat_3d
                feat_3d = feat_3d_new
                mask_chunk = torch.ones_like(
                    mask_chunk
                )  # every point needs to be evaluted
        elif len(processed_data.keys()) > 2:  # legacy, for old processed features
            feat_3d, mask_visible, mask_chunk = (
                processed_data["feat"],  # N_feat, 512, 1 (20000, 512, 1)
                processed_data["mask"],  # index of visible points in N_feat
                processed_data["mask_full"],  # N_pt
            )
            mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
            mask[mask_visible] = (
                True  # mask out points without feature assigned  Shpae: N_feat
            )

        if len(feat_3d.shape) > 2:
            feat_3d = feat_3d[..., 0]  # N_feat, 512

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in  # N_pt, 3

        # calculate the corresponding point features after voxelization
        if self.split == "train" and flag_mask_merge:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True
            )
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind]  # voxelized visible mask for entire point cloud
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1 != -1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]
        elif (
            self.split == "train" and not flag_mask_merge
        ):  # legacy, for old processed features
            feat_3d = feat_3d[mask]  # 2d feats Shape: N_vis, C
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True
            )  # (N_vpt, 3), (N_vpt, 3), (N_vpt,), (N_pt), (N_vpt)
            mask_chunk[mask_chunk.clone()] = mask  # 2d feat points mask in all space
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[
                vox_ind
            ]  # 2d feat points mask voxelized space Shape: N_vpt
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = (
                mask_ind  # same as mask_chunk but with -1 or point_idx as values
            )

            index1 = index1[vox_ind]  # -1 or point_idx among all points in voxelized
            chunk_ind = index1[
                index1 != -1
            ]  # 2d feat point_idx among all points in voxelized

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1  # =mask_chunk
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]  # N_vvis, C
        else:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs[mask_chunk],
                feats_in[mask_chunk],
                labels_in[mask_chunk],
                return_ind=True,
            )
            vox_ind = torch.from_numpy(vox_ind)
            feat_3d = feat_3d[vox_ind]
            mask = mask[vox_ind]

        if self.eval_all:  # during evaluation, no voxelization for GT labels
            labels = labels_in

        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()  # (N_vpt, 3)
        coords = torch.cat(
            (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1
        )  # (N_vpt, 4)
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.0
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            return (
                coords,
                feats,
                labels,
                feat_3d,
                mask,
                torch.from_numpy(inds_reconstruct).long(),
            )
        return coords, feats, labels, feat_3d, mask


class FusedFeatureLoader_strat(Point3DLoader):
    """Dataloader for fused point features."""

    def __init__(
        self,
        datapath_prefix,
        datapath_prefix_feat,
        voxel_size=0.05,
        voxel_max=None,
        split="train",
        aug=False,
        memcache_init=False,
        identifier=7791,
        loop=1,
        eval_all=False,
        input_color=False,
    ):
        super().__init__(
            datapath_prefix=datapath_prefix,
            voxel_size=voxel_size,
            split=split,
            aug=aug,
            memcache_init=memcache_init,
            identifier=identifier,
            loop=loop,
            eval_all=eval_all,
            input_color=input_color,
        )
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.aug = aug
        self.input_color = (
            input_color  # decide whether we use point color values as input
        )

        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        train_transform = None
        from dataset import transform

        if self.aug:
            self.train_transform = transform.Compose(
                [
                    transform.RandomRotate(along_z=True),
                    transform.RandomScale(
                        scale_low=0.8,
                        scale_high=1.2,
                    ),
                    transform.RandomDropColor(color_augment=0.0),
                ]
            )

        # Precompute the occurances for each scene
        # for training sets, ScanNet and Matterport has 5 each, nuscene 1
        # for evaluation/test sets, all has just one
        if "nuscenes" in self.dataset_name:  # only one file for each scene
            self.list_occur = None
        else:
            self.list_occur = []
            for data_path in self.data_paths:
                if "scannet" in self.dataset_name:
                    scene_name = data_path[:-15].split("/")[-1]
                else:
                    scene_name = data_path[:-4].split("/")[-1]
                file_dirs = glob(join(self.datapath_feat, scene_name + "_*.pt"))
                self.list_occur.append(len(file_dirs))
            # some scenes in matterport have no features at all
            ind = np.where(np.array(self.list_occur) != 0)[0]
            if np.any(np.array(self.list_occur) == 0):
                data_paths, list_occur = [], []
                for i in ind:
                    data_paths.append(self.data_paths[i])
                    list_occur.append(self.list_occur[i])
                self.data_paths = data_paths
                self.list_occur = list_occur

        if len(self.data_paths) == 0:
            raise Exception("0 file is loaded in the feature loader.")

    def __getitem__(self, index_long):
        index = index_long % len(self.data_paths)

        # (N_pt, 3), (N_pt, 3), (N_pt,)
        locs_in, feats_in, labels_in = torch.load(self.data_paths[index])

        labels_in[labels_in == -100] = 255
        labels_in = labels_in.astype(np.uint8)
        if np.isscalar(feats_in) and feats_in == 0:
            # no color in the input point cloud, e.g nuscenes lidar
            feats_in = np.zeros_like(locs_in)
        else:
            feats_in = (feats_in + 1.0) * 127.5

        # load 3D features
        if self.dataset_name == "scannet_3d":
            scene_name = self.data_paths[index][:-15].split("/")[-1]
        else:
            scene_name = self.data_paths[index][:-4].split("/")[-1]

        if "nuscenes" not in self.dataset_name:
            n_occur = self.list_occur[index]
            if n_occur > 1:
                nn_occur = np.random.randint(n_occur)
            elif n_occur == 1:
                nn_occur = 0
            else:
                raise NotImplementedError

            processed_data = torch.load(
                join(self.datapath_feat, scene_name + "_%d.pt" % (nn_occur))
            )
        else:
            # no repeated file
            processed_data = torch.load(join(self.datapath_feat, scene_name + ".pt"))

        flag_mask_merge = False
        if len(processed_data.keys()) == 2:
            flag_mask_merge = True
            feat_3d, mask_chunk = (
                processed_data["feat"],
                processed_data["mask_full"],
            )
            if isinstance(
                mask_chunk, np.ndarray
            ):  # if the mask itself is a numpy array
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
            if self.split != "train":  # val or test set
                feat_3d_new = torch.zeros(
                    (locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype
                )
                feat_3d_new[mask] = feat_3d
                feat_3d = feat_3d_new
                mask_chunk = torch.ones_like(
                    mask_chunk
                )  # every point needs to be evaluted
        elif len(processed_data.keys()) > 2:  # legacy, for old processed features
            feat_3d, mask_visible, mask_chunk = (
                processed_data["feat"],  # N_feat, 512, 1 (20000, 512, 1)
                processed_data["mask"],  # index of visible points in N_feat
                processed_data["mask_full"],  # N_pt
            )
            mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
            mask[mask_visible] = (
                True  # mask out points without feature assigned  Shpae: N_feat
            )

        if len(feat_3d.shape) > 2:
            feat_3d = feat_3d[..., 0]  # N_feat, 512

        # calculate the corresponding point features after voxelization
        if self.split == "train":
            if not flag_mask_merge:
                feat_3d = feat_3d[mask]  # 2d feats Shape: N_vis, C
                mask_chunk[mask_chunk.clone()] = (
                    mask  # 2d feat points mask in all space
                )
            locs, feats, labels, vox_ind = feat_point_voxelize(
                locs_in,
                feats_in,
                labels_in,
                voxel_size=self.voxel_size,
                voxel_max=self.voxel_max,
            )  # (N_vpt, 3), (N_vpt, 3), (N_vpt,), (N_pt), (N_vpt)

            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[
                vox_ind
            ]  # 2d feat points mask voxelized space Shape: N_vpt

            # Find the indices of 2d-feat points among all points
            mask_ind = torch.nonzero(mask_chunk).squeeze()
            index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = (
                mask_ind  # same as mask_chunk but with -1 or point_idx as values
            )

            index1 = index1[
                vox_ind
            ]  # -1 or point_idx among all points in voxelized (sorted by vox_ind so 1-1 relation to feats)
            chunk_ind = index1[
                index1 != -1
            ]  # 2d feat point_idx among all points in voxelized

            index3 = torch.cumsum(mask_chunk, dim=0, dtype=int)
            # 2d feature point_idx after voxelization among all 2d feature points
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]  # N_vvis, C
        else:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs[mask_chunk],
                feats_in[mask_chunk],
                labels_in[mask_chunk],
                return_ind=True,
            )
            vox_ind = torch.from_numpy(vox_ind)
            feat_3d = feat_3d[vox_ind]
            mask = mask[vox_ind]

        if self.eval_all:  # during evaluation, no voxelization for GT labels
            labels = labels_in

        if self.aug:
            locs, feats = self.train_transform(locs, feats)

        coords = torch.from_numpy(locs)  # (N_vpt, 3)

        coord_min = torch.min(coords, 0)[0]
        coords -= coord_min

        coords = torch.cat(
            (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1
        )  # (N_vpt, 4)

        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.0
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        return coords, feats, labels, feat_3d, mask


def collation_fn(batch):
    """
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    """
    coords, feats, labels, feat_3d, mask_chunk = list(zip(*batch))
    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return (
        torch.cat(coords),
        torch.cat(feats),
        torch.cat(labels),
        torch.cat(feat_3d),
        torch.cat(mask_chunk),
    )


def collate_fn_limit(batch, max_batch_points, logger):
    """
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    """
    coords, feats, labels, feat_3d, mask_chunk = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    offset, count = [], 0
    k = 0
    for item in coords:
        count += item.shape[0]
        if count > max_batch_points:
            break
        k += 1
        offset.append(count)

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in coords])
        s_now = sum([x.shape[0] for x in coords[:k]])
        logger.warning(
            "batch_size shortened from {} to {}, points from {} to {}".format(
                len(batch), k, s, s_now
            )
        )

    return (
        torch.cat(coords[:k]),
        torch.cat(feats[:k]),
        torch.cat(labels[:k]),
        torch.cat(feat_3d[:k]),
        torch.cat(mask_chunk[:k]),
        torch.IntTensor(offset[:k]),
    )


def collation_fn_eval_all(batch):
    """
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    """
    coords, feats, labels, feat_3d, mask, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return (
        torch.cat(coords),
        torch.cat(feats),
        torch.cat(labels),
        torch.cat(feat_3d),
        torch.cat(mask),
        torch.cat(inds_recons),
    )
