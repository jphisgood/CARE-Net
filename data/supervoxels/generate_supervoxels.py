# """
# Modified from Ouyang et al.
# https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation
# """

# import os
# import SimpleITK as sitk
# import glob
# from skimage.measure import label
# import scipy.ndimage.morphology as snm
# from felzenszwalb_3d import *

# base_dir = '/root/rivermind-data/Geoproto-main/data/ABD/ABDOMEN_CT/sabs_CT_normalized'
# # base_dir = '<path_to_data>/CMR/cmr_MR_normalized'

# imgs = glob.glob(os.path.join(base_dir, 'image*'))
# labels = glob.glob(os.path.join(base_dir, 'label*'))

# imgs = sorted(imgs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
# labels = sorted(labels, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

# fg_thresh = 10

# MODE = 'MIDDLE'
# n_sv = 5000
# # n_sv = 1000

# def read_nii_bysitk(input_fid):
#     """ read nii to numpy through simpleitk
#         peelinfo: taking direction, origin, spacing and metadata out
#     """
#     img_obj = sitk.ReadImage(input_fid)
#     img_np = sitk.GetArrayFromImage(img_obj)
#     return img_np

# # thresholding the intensity values to get a binary mask of the patient
# def fg_mask2d(img_2d, thresh):
#     mask_map = np.float32(img_2d > thresh)
#     def getLargestCC(segmentation):  # largest connected components
#         labels = label(segmentation)
#         assert (labels.max() != 0)  # assume at least 1 CC
#         largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
#         return largestCC

#     if mask_map.max() < 0.999:
#         return mask_map
#     else:
#         post_mask = getLargestCC(mask_map)
#         fill_mask = snm.binary_fill_holes(post_mask)
#     return fill_mask


# # remove supervoxels within the empty regions
# def supervox_masking(seg, mask):

#     seg[seg == 0] = seg.max() + 1
#     seg = np.int32(seg)
#     seg[mask == 0] = 0

#     return seg

# # make supervoxels
# for img_path in imgs:
#     img = read_nii_bysitk(img_path)
#     img = 255 * (img - img.min()) / img.ptp()

#     reader = sitk.ImageFileReader()
#     reader.SetFileName(img_path)
#     reader.LoadPrivateTagsOn()
#     reader.ReadImageInformation()

#     x = float(reader.GetMetaData('pixdim[1]'))
#     y = float(reader.GetMetaData('pixdim[2]'))
#     z = float(reader.GetMetaData('pixdim[3]'))

#     segments_felzenszwalb = felzenszwalb_3d(img, min_size=n_sv, sigma=0, spacing=(z, x, y))

#     # post processing: remove bg (low intensity regions)
#     fg_mask_vol = np.zeros(segments_felzenszwalb.shape)
#     for ii in range(segments_felzenszwalb.shape[0]):
#         _fgm = fg_mask2d(img[ii, ...], fg_thresh)
#         fg_mask_vol[ii] = _fgm
#     processed_seg_vol = supervox_masking(segments_felzenszwalb, fg_mask_vol)

#     # write to nii.gz
#     out_seg = sitk.GetImageFromArray(processed_seg_vol)

#     idx = os.path.basename(img_path).split("_")[-1].split(".nii.gz")[0]

#     # seg_fid = os.path.join(f'<path_to_data>/supervoxels_{n_sv}/', f'superpix-{MODE}_{idx}.nii.gz')
#     # sitk.WriteImage(out_seg, seg_fid)
#     print(f'image with id {idx} has finished')


"""
Generate 3-D supervoxels for GeoProto abdominal CT and MRI datasets.

Run from:
    /root/rivermind-data/Geoproto-main/data/supervoxels

Examples:
    python generate_supervoxels.py --dataset ct --clean
    python generate_supervoxels.py --dataset mri --clean
    python generate_supervoxels.py --dataset all --clean
"""

import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label

from felzenszwalb_3d import felzenszwalb_3d


DATASETS = {
    "ct": {
        "name": "SABS CT",
        "input_dir": "/root/rivermind-data/Geoproto-main/data/ABD/ABDOMEN_CT/sabs_CT_normalized",
        "output_dir": "/root/rivermind-data/Geoproto-main/data/ABD/ABDOMEN_CT/supervoxels_5000",
    },
    "mri": {
        "name": "CHAOST2 MRI",
        "input_dir": "/root/rivermind-data/Geoproto-main/data/ABD/ABDOMEN_MR/chaos_MR_T2_normalized",
        "output_dir": "/root/rivermind-data/Geoproto-main/data/ABD/ABDOMEN_MR/supervoxels_5000",
    },
}

FG_THRESH = 10
MODE = "MIDDLE"
N_SUPERVOXEL = 5000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("ct", "mri", "all"),
        default="all",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete old superpix-MIDDLE_*.nii.gz files first.",
    )
    return parser.parse_args()


def numeric_id(path):
    name = os.path.basename(path)
    return int(name.split("_")[-1].split(".nii.gz")[0])


def read_nii(path):
    image_obj = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image_obj)
    return image_array, image_obj


def normalize_to_255(image):
    image = image.astype(np.float32)
    value_range = float(np.ptp(image))
    if value_range <= 0:
        return np.zeros_like(image, dtype=np.float32)
    return 255.0 * (image - float(image.min())) / value_range


def fg_mask2d(image_2d, threshold):
    mask_map = np.asarray(image_2d > threshold, dtype=np.float32)

    if mask_map.max() < 0.999:
        return mask_map

    connected = label(mask_map)
    if connected.max() == 0:
        return mask_map

    component_sizes = np.bincount(connected.ravel())
    component_sizes[0] = 0
    largest_component_id = int(np.argmax(component_sizes))
    largest_component = connected == largest_component_id

    return binary_fill_holes(largest_component)


def supervoxel_masking(segmentation, foreground_mask):
    segmentation = segmentation.copy()
    segmentation[segmentation == 0] = segmentation.max() + 1
    segmentation = segmentation.astype(np.int32)
    segmentation[foreground_mask == 0] = 0
    return segmentation


def clean_output(output_dir):
    pattern = os.path.join(output_dir, f"superpix-{MODE}_*.nii.gz")
    old_files = glob.glob(pattern)
    for path in old_files:
        os.remove(path)
    print(f"Removed {len(old_files)} old files from {output_dir}")


def generate_dataset(dataset_key, clean=False):
    cfg = DATASETS[dataset_key]
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    if clean:
        clean_output(output_dir)

    image_files = sorted(
        glob.glob(os.path.join(input_dir, "image_*.nii.gz")),
        key=numeric_id,
    )

    if not image_files:
        raise FileNotFoundError(
            f"No image_*.nii.gz files found in: {input_dir}"
        )

    print("\n" + "=" * 80)
    print(f"Dataset:     {cfg['name']}")
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Image count: {len(image_files)}")
    print(f"MODE:        {MODE}")
    print(f"min_size:    {N_SUPERVOXEL}")
    print("=" * 80)

    for index, image_path in enumerate(image_files, start=1):
        scan_id = numeric_id(image_path)
        image, reference_image = read_nii(image_path)
        image_normalized = normalize_to_255(image)

        spacing_x, spacing_y, spacing_z = reference_image.GetSpacing()
        # Preserve the original repository's spacing order: (z, x, y).
        spacing_zyx = (
            float(spacing_z),
            float(spacing_x),
            float(spacing_y),
        )

        print(
            f"[{index:02d}/{len(image_files):02d}] "
            f"image_{scan_id}.nii.gz "
            f"shape={image.shape}, spacing_zyx={spacing_zyx}"
        )

        segments = felzenszwalb_3d(
            image_normalized,
            min_size=N_SUPERVOXEL,
            sigma=0,
            spacing=spacing_zyx,
        )

        foreground_mask = np.zeros(
            segments.shape,
            dtype=np.uint8,
        )

        for z_index in range(segments.shape[0]):
            foreground_mask[z_index] = fg_mask2d(
                image_normalized[z_index],
                FG_THRESH,
            ).astype(np.uint8)

        processed_segmentation = supervoxel_masking(
            segments,
            foreground_mask,
        )

        output_image = sitk.GetImageFromArray(
            processed_segmentation.astype(np.int32)
        )
        output_image.CopyInformation(reference_image)

        output_path = os.path.join(
            output_dir,
            f"superpix-{MODE}_{scan_id}.nii.gz",
        )
        sitk.WriteImage(output_image, output_path)

        print(f"    Saved -> {output_path}")

    generated_files = glob.glob(
        os.path.join(output_dir, f"superpix-{MODE}_*.nii.gz")
    )

    print(f"Finished {cfg['name']}: {len(generated_files)} files generated.")


def main():
    args = parse_args()

    dataset_keys = (
        ("ct", "mri")
        if args.dataset == "all"
        else (args.dataset,)
    )

    for dataset_key in dataset_keys:
        generate_dataset(dataset_key, clean=args.clean)


if __name__ == "__main__":
    main()