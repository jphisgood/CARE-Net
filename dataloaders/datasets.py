"""
Dataset for GeoProto training.

Extended from the official GeoProto/ADNet loader with support for:
    data/Prostate/NCI/NCI_normalized
    data/Prostate/NCI/supervoxels_5000
    data/Prostate/UCLH/UCLH_normalized
    data/Prostate/UCLH/supervoxels_5000
"""

import glob
import os
import random
import re

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import torchvision.transforms as deftfx

from . import image_transforms as myit


_DATASET_LAYOUT = {
    "CARDIAC_bssFP": "cmr_bssFP_normalized",
    "CARDIAC_LGE": "cmr_LGE_normalized",
    "ABDOMEN_MR": "chaos_MR_T2_normalized",
    "ABDOMEN_CT": "sabs_CT_normalized",
    "Prostate_NCI": "NCI_normalized",
    "Prostate_UCLH": "UCLH_normalized",
}


def _strip_nii_suffix(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return os.path.splitext(filename)[0]


def extract_case_id(path):
    """Extract the trailing numeric case ID from all supported filenames.

    Examples:
        image_1.nii.gz               -> "1"
        label_1.nii.gz               -> "1"
        superpix-MIDDLE_1.nii.gz     -> "1"
        supervoxel_001.nii.gz        -> "1"
    """
    stem = _strip_nii_suffix(os.path.basename(path))
    match = re.search(r"(\d+)$", stem)
    if match is None:
        raise ValueError(
            "Cannot extract a trailing numeric case ID from: {}".format(path)
        )
    return str(int(match.group(1)))


def _numeric_case_sort(case_id):
    try:
        return 0, int(case_id)
    except ValueError:
        return 1, case_id


def _build_unique_map(paths, kind):
    result = {}
    for path in paths:
        case_id = extract_case_id(path)
        if case_id in result:
            raise RuntimeError(
                "Duplicate {} files for case {}: {} and {}".format(
                    kind, case_id, result[case_id], path
                )
            )
        result[case_id] = path
    return result


def _resolve_dataset_directories(dataset, data_dir, n_sv):
    if dataset not in _DATASET_LAYOUT:
        raise ValueError(
            "Unsupported training dataset {!r}. Supported values: {}".format(
                dataset, sorted(_DATASET_LAYOUT)
            )
        )

    data_dir = os.path.abspath(os.path.normpath(data_dir))
    normalized_name = _DATASET_LAYOUT[dataset]

    # Accept either the institution/dataset parent directory or the normalized
    # directory itself. This keeps the original GeoProto paths compatible.
    if os.path.basename(data_dir) == normalized_name:
        normalized_dir = data_dir
        dataset_root = os.path.dirname(data_dir)
    else:
        normalized_dir = os.path.join(data_dir, normalized_name)
        dataset_root = data_dir

    supervoxel_dir = os.path.join(dataset_root, "supervoxels_{}".format(n_sv))

    if not os.path.isdir(normalized_dir):
        raise FileNotFoundError(
            "Normalized data directory not found: {}".format(normalized_dir)
        )
    if not os.path.isdir(supervoxel_dir):
        raise FileNotFoundError(
            "Supervoxel directory not found: {}".format(supervoxel_dir)
        )

    return normalized_dir, supervoxel_dir


def _collect_case_triplets(dataset, data_dir, n_sv):
    normalized_dir, supervoxel_dir = _resolve_dataset_directories(
        dataset, data_dir, n_sv
    )

    image_paths = glob.glob(os.path.join(normalized_dir, "image*.nii.gz"))
    label_paths = glob.glob(os.path.join(normalized_dir, "label*.nii.gz"))
    super_paths = glob.glob(os.path.join(supervoxel_dir, "super*.nii.gz"))

    image_map = _build_unique_map(image_paths, "image")
    label_map = _build_unique_map(label_paths, "label")
    super_map = _build_unique_map(super_paths, "supervoxel")

    image_ids = set(image_map)
    label_ids = set(label_map)
    super_ids = set(super_map)

    missing_labels = sorted(image_ids - label_ids, key=_numeric_case_sort)
    missing_super = sorted(image_ids - super_ids, key=_numeric_case_sort)
    extra_labels = sorted(label_ids - image_ids, key=_numeric_case_sort)
    extra_super = sorted(super_ids - image_ids, key=_numeric_case_sort)

    if missing_labels or missing_super or extra_labels or extra_super:
        raise RuntimeError(
            "Dataset file IDs do not match. missing_labels={}, missing_super={}, "
            "extra_labels={}, extra_super={}".format(
                missing_labels[:20],
                missing_super[:20],
                extra_labels[:20],
                extra_super[:20],
            )
        )

    case_ids = sorted(image_ids, key=_numeric_case_sort)
    if not case_ids:
        raise RuntimeError(
            "No matched image/label/supervoxel cases found under {}".format(
                normalized_dir
            )
        )

    triplets = [
        (image_map[case_id], label_map[case_id], super_map[case_id])
        for case_id in case_ids
    ]

    print("###### Dataset paths ######")
    print("dataset       : {}".format(dataset))
    print("normalized    : {}".format(normalized_dir))
    print("supervoxels   : {}".format(supervoxel_dir))
    print("matched cases : {}".format(len(triplets)))

    return triplets


class TrainDataset(Dataset):

    def __init__(self, args):
        self.n_shot = args["n_shot"]
        self.n_way = args["n_way"]
        self.n_query = args["n_query"]
        self.n_sv = args["n_sv"]
        self.max_iter = args["max_iter"]
        self.read = True
        self.train_sampling = "neighbors"

        self.min_size = args["min_size"]
        self.test_label = args["test_label"]
        self.exclude_label = args["exclude_label"]
        self.use_gt = args["use_gt"]

        triplets = _collect_case_triplets(
            dataset=args["dataset"],
            data_dir=args["data_dir"],
            n_sv=args["n_sv"],
        )
        self.image_dirs = [item[0] for item in triplets]
        self.label_dirs = [item[1] for item in triplets]
        self.sprvxl_dirs = [item[2] for item in triplets]

        if self.read:
            self.images = {}
            self.labels = {}
            self.sprvxls = {}
            for image_dir, label_dir, sprvxl_dir in triplets:
                image = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
                label = sitk.GetArrayFromImage(sitk.ReadImage(label_dir))
                supervoxel = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir))

                if image.shape != label.shape or image.shape != supervoxel.shape:
                    raise ValueError(
                        "Shape mismatch for case {}: image={}, label={}, supervoxel={}".
                        format(
                            extract_case_id(image_dir),
                            image.shape,
                            label.shape,
                            supervoxel.shape,
                        )
                    )

                self.images[image_dir] = image
                self.labels[label_dir] = label
                self.sprvxls[sprvxl_dir] = supervoxel

    def __len__(self):
        return self.max_iter

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = img.max() - cmin + 1e-5
        img = img - cmin + 1e-5
        img = irange * np.power(img / irange, gamma)
        return img + cmin

    def geom_transform(self, img, mask, supervoxel):
        affine = {
            "rotate": 5,
            "shift": (5, 5),
            "shear": 5,
            "scale": (0.9, 1.2),
        }
        transform = deftfx.Compose(
            [
                myit.RandomAffine(
                    affine.get("rotate"),
                    affine.get("shift"),
                    affine.get("shear"),
                    affine.get("scale"),
                    affine.get("scale_iso", True),
                    order=3,
                ),
                myit.ElasticTransform(10, 5),
            ]
        )

        if len(img.shape) > 4:
            n_shot = img.shape[1]
            for shot in range(n_shot):
                cat_img_mask = np.concatenate(
                    (img[0, shot], mask[:, shot])
                ).transpose(1, 2, 0)
                cat_supervoxel = supervoxel[shot].transpose(1, 2, 0)
                cat_img_mask = transform(cat_img_mask).transpose(2, 0, 1)
                cat_supervoxel = transform(cat_supervoxel).transpose(2, 0, 1)
                img[0, shot] = cat_img_mask[:3]
                mask[:, shot] = np.rint(cat_img_mask[3:])
                supervoxel[shot] = np.rint(cat_supervoxel)
        else:
            for query_index in range(img.shape[0]):
                cat_img_mask = np.concatenate(
                    (img[query_index], mask[query_index][None])
                ).transpose(1, 2, 0)
                cat_supervoxel = supervoxel[query_index].transpose(1, 2, 0)
                cat_img_mask = transform(cat_img_mask).transpose(2, 0, 1)
                cat_supervoxel = transform(cat_supervoxel).transpose(2, 0, 1)
                img[query_index] = cat_img_mask[:3]
                mask[query_index] = np.rint(cat_img_mask[3:].squeeze())
                supervoxel[query_index] = np.rint(cat_supervoxel)

        return img, mask, supervoxel

    def __getitem__(self, idx):
        del idx
        pat_idx = random.choice(range(len(self.image_dirs)))

        if self.read:
            img = self.images[self.image_dirs[pat_idx]].copy()
            gt = self.labels[self.label_dirs[pat_idx]].copy()
            sprvxl = self.sprvxls[self.sprvxl_dirs[pat_idx]].copy()
        else:
            img = sitk.GetArrayFromImage(sitk.ReadImage(self.image_dirs[pat_idx]))
            gt = sitk.GetArrayFromImage(sitk.ReadImage(self.label_dirs[pat_idx]))
            sprvxl = sitk.GetArrayFromImage(sitk.ReadImage(self.sprvxl_dirs[pat_idx]))

        if self.exclude_label is not None:
            slice_indices = np.arange(gt.shape[0])
            exclude_mask = np.full(gt.shape[0], True, dtype=bool)
            for label_id in self.exclude_label:
                exclude_mask = exclude_mask & (
                    np.sum(gt == label_id, axis=(1, 2)) > 0
                )
            exclude_idx = slice_indices[exclude_mask]
        else:
            exclude_idx = []

        img = (img - img.mean()) / (img.std() + 1e-8)

        lbl = gt.copy() if self.use_gt else sprvxl.copy()
        whole_supervoxel = lbl

        unique = list(np.unique(lbl))
        if 0 in unique:
            unique.remove(0)
        if self.use_gt:
            unique = list(set(unique) - set(self.test_label))
        if not unique:
            raise RuntimeError("No eligible training labels were found in the sampled volume.")

        size = 0
        while size < self.min_size:
            required = self.n_shot * self.n_way + self.n_query
            n_slices = required - 1
            while n_slices < required:
                cls_idx = random.choice(unique)
                sli_idx = np.where(np.sum(lbl == cls_idx, axis=(1, 2)) > 0)[0]
                sli_idx = list(
                    set(sli_idx) - set(np.intersect1d(sli_idx, exclude_idx))
                )
                sli_idx.sort()
                n_slices = len(sli_idx)

            subsets = []
            for slice_id in sli_idx:
                if not subsets or subsets[-1][-1] + 1 != slice_id:
                    subsets.append([slice_id])
                else:
                    subsets[-1].append(slice_id)
            subsets = [subset for subset in subsets if len(subset) >= required]

            if not subsets:
                return self.__getitem__(0)

            subset = random.choice(subsets)
            start = random.choice(subset[: -(required - 1)])
            sample = np.arange(start, start + required)
            lbl_cls = (lbl == cls_idx).astype(np.int32)
            size = max(np.sum(lbl_cls[sample[0]]), np.sum(lbl_cls[sample[1]]))

        if np.random.random(1) > 0.5:
            sample = sample[::-1]

        support_count = self.n_shot * self.n_way
        sup_lbl = lbl_cls[sample[:support_count]][None]
        qry_lbl = lbl_cls[sample[support_count:]]
        whole_supervoxel = whole_supervoxel[sample[:support_count]][None]

        sup_img = img[sample[:support_count]][None]
        sup_img = np.stack((sup_img, sup_img, sup_img), axis=2)
        qry_img = img[sample[support_count:]]
        qry_img = np.stack((qry_img, qry_img, qry_img), axis=1)
        superv = whole_supervoxel

        if np.random.random(1) > 0.5:
            qry_img = self.gamma_tansform(qry_img)
        else:
            sup_img = self.gamma_tansform(sup_img)

        if np.random.random(1) > 0.5:
            qry_img, qry_lbl, _ = self.geom_transform(
                qry_img,
                qry_lbl,
                whole_supervoxel.astype(np.float64),
            )
        else:
            sup_img, sup_lbl, superv = self.geom_transform(
                sup_img,
                sup_lbl,
                whole_supervoxel.astype(np.float64),
            )

        return {
            "support_images": sup_img,
            "support_fg_labels": sup_lbl,
            "query_images": qry_img,
            "query_labels": qry_lbl,
            "selected_class": cls_idx,
            "whole_supervoxel": superv,
        }
