"""Utility functions used by GeoProto."""

import operator
import random

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_bbox(fg_mask, inst_mask):
    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)
    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min, y_max = mask_idx[0].min(), mask_idx[0].max()
    x_min, x_max = mask_idx[1].min(), mask_idx[1].max()
    fg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 1

    for instance_id in cls_ids:
        mask_idx = np.where(inst_mask[0] == instance_id)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 0
    return fg_bbox, bg_bbox


def t2n(img_t):
    return img_t.data.cpu().numpy() if img_t.is_cuda else img_t.data.numpy()


def to01(x_np):
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)


def compose_wt_simple(is_wce, data_name):
    supported = {
        "SABS",
        "SABS_Superpix",
        "C0",
        "C0_Superpix",
        "CHAOST2",
        "CHAOST2_Superpix",
        "CARDIAC_bssFP",
        "CARDIAC_LGE",
        "Prostate_NCI",
        "Prostate_UCLH",
    }
    if is_wce:
        if data_name not in supported:
            raise NotImplementedError(
                "Weighted CE is not configured for dataset: {}".format(data_name)
            )
        return torch.FloatTensor([0.05, 1.0]).cuda()
    return torch.FloatTensor([1.0, 1.0]).cuda()


class CircularList(list):
    def __getitem__(self, x):
        if isinstance(x, slice):
            return [self[index] for index in self._rangeify(x)]
        index = operator.index(x)
        try:
            return super().__getitem__(index % len(self))
        except ZeroDivisionError:
            raise IndexError("list index out of range")

    def _rangeify(self, slice_value):
        start, stop, step = slice_value.start, slice_value.stop, slice_value.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        return range(start, stop, step)


def compute_sdf_from_mask(masks, K_bins=10):
    """Convert binary support masks to ordinal EDT-bin labels."""
    number_of_masks, height, width = masks.shape
    sdf_maps = torch.zeros(number_of_masks, height, width, dtype=torch.float32)

    for index in range(number_of_masks):
        mask_np = masks[index].detach().cpu().numpy().astype(np.uint8)
        if mask_np.sum() == 0:
            continue
        distance = distance_transform_edt(mask_np).astype(np.float32)
        max_distance = distance.max()
        if max_distance > 0:
            scale = np.floor(
                distance / max_distance * (K_bins - 1)
            ).astype(np.int32)
            scale = np.clip(scale, 0, K_bins - 1)
        else:
            scale = np.zeros_like(distance, dtype=np.int32)
        scale = scale * mask_np
        sdf_maps[index] = torch.from_numpy(scale.astype(np.float32))

    return sdf_maps.unsqueeze(1)
