"""
Generate GeoProto supervoxels_5000 for Cardiac b-SSFP and LGE.

Run this file from:
    /root/rivermind-data/Geoproto-main/data/supervoxels

Examples:
    python generate_supervoxels_cardiac.py --dataset bssfp --clean
    python generate_supervoxels_cardiac.py --dataset lge --clean
    python generate_supervoxels_cardiac.py --dataset all --clean
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
    "bssfp": {
        "name": "Cardiac b-SSFP",
        "input_dir": (
            "/root/rivermind-data/Geoproto-main/"
            "data/Cardiac/bSSFP/cmr_bssFP_normalized"
        ),
        "output_dir": (
            "/root/rivermind-data/Geoproto-main/"
            "data/Cardiac/bSSFP/supervoxels_5000"
        ),
    },
    "lge": {
        "name": "Cardiac LGE",
        "input_dir": (
            "/root/rivermind-data/Geoproto-main/"
            "data/Cardiac/LGE/cmr_LGE_normalized"
        ),
        "output_dir": (
            "/root/rivermind-data/Geoproto-main/"
            "data/Cardiac/LGE/supervoxels_5000"
        ),
    },
}

FG_THRESH = 10
MODE = "MIDDLE"
N_SV = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GeoProto cardiac supervoxels_5000."
    )
    parser.add_argument(
        "--dataset",
        choices=("bssfp", "lge", "all"),
        default="all",
        help="Dataset to process. Default: all.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing superpix-MIDDLE_*.nii.gz files first.",
    )
    return parser.parse_args()


def scan_id(path):
    name = os.path.basename(path)
    return int(name.split("_")[-1].split(".nii.gz")[0])


def normalize_to_255(volume):
    volume = volume.astype(np.float32)
    value_range = float(np.ptp(volume))
    if value_range <= 0:
        return np.zeros_like(volume, dtype=np.float32)
    return 255.0 * (volume - float(volume.min())) / value_range


def foreground_mask_2d(image_2d, threshold):
    mask = np.asarray(image_2d > threshold, dtype=np.float32)

    if mask.max() < 0.999:
        return mask

    components = label(mask)
    if components.max() == 0:
        return mask

    component_sizes = np.bincount(components.ravel())
    component_sizes[0] = 0
    largest_component = components == int(np.argmax(component_sizes))

    return binary_fill_holes(largest_component)


def mask_supervoxels(segmentation, foreground_mask):
    segmentation = segmentation.copy()
    segmentation[segmentation == 0] = segmentation.max() + 1
    segmentation = segmentation.astype(np.int32)
    segmentation[foreground_mask == 0] = 0
    return segmentation


def clean_output(output_dir):
    old_files = glob.glob(
        os.path.join(output_dir, f"superpix-{MODE}_*.nii.gz")
    )
    for path in old_files:
        os.remove(path)
    print(f"Removed {len(old_files)} old files from: {output_dir}")


def generate_dataset(dataset_key, clean=False):
    cfg = DATASETS[dataset_key]
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found:\n{input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    if clean:
        clean_output(output_dir)

    image_files = sorted(
        glob.glob(os.path.join(input_dir, "image_*.nii.gz")),
        key=scan_id,
    )

    label_files = sorted(
        glob.glob(os.path.join(input_dir, "label_*.nii.gz")),
        key=scan_id,
    )

    if not image_files:
        raise FileNotFoundError(
            f"No image_*.nii.gz files found in:\n{input_dir}"
        )

    if len(image_files) != len(label_files):
        raise RuntimeError(
            f"Image/label count mismatch in {input_dir}: "
            f"{len(image_files)} images vs {len(label_files)} labels"
        )

    print("\n" + "=" * 80)
    print(f"Dataset:       {cfg['name']}")
    print(f"Input:         {input_dir}")
    print(f"Output:        {output_dir}")
    print(f"Image count:   {len(image_files)}")
    print(f"Label count:   {len(label_files)}")
    print(f"MODE:          {MODE}")
    print(f"n_sv:          {N_SV}")
    print(f"FG threshold:  {FG_THRESH}")
    print("=" * 80)

    for index, image_path in enumerate(image_files, start=1):
        current_id = scan_id(image_path)

        reference_image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(reference_image)
        image_normalized = normalize_to_255(image)

        spacing_x, spacing_y, spacing_z = reference_image.GetSpacing()

        # Match the original GeoProto implementation: NumPy order is Z,Y,X,
        # while the repository passes spacing as (z, x, y).
        spacing_for_sv = (
            float(spacing_z),
            float(spacing_x),
            float(spacing_y),
        )

        print(
            f"[{index:02d}/{len(image_files):02d}] "
            f"image_{current_id}.nii.gz "
            f"shape={image.shape}, spacing={spacing_for_sv}"
        )

        segments = felzenszwalb_3d(
            image_normalized,
            min_size=N_SV,
            sigma=0,
            spacing=spacing_for_sv,
        )

        foreground_volume = np.zeros(
            segments.shape,
            dtype=np.uint8,
        )

        for z_index in range(segments.shape[0]):
            foreground_volume[z_index] = foreground_mask_2d(
                image_normalized[z_index],
                FG_THRESH,
            ).astype(np.uint8)

        processed_segments = mask_supervoxels(
            segments,
            foreground_volume,
        )

        output_image = sitk.GetImageFromArray(
            processed_segments.astype(np.int32)
        )
        output_image.CopyInformation(reference_image)

        output_path = os.path.join(
            output_dir,
            f"superpix-{MODE}_{current_id}.nii.gz",
        )
        sitk.WriteImage(output_image, output_path)

        print(
            f"    Saved -> {output_path}\n"
            f"    Unique labels = {len(np.unique(processed_segments))}"
        )

    generated = glob.glob(
        os.path.join(output_dir, f"superpix-{MODE}_*.nii.gz")
    )

    print("\n" + "-" * 80)
    print(f"Finished {cfg['name']}")
    print(f"Generated files: {len(generated)}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)


def main():
    args = parse_args()

    selected = (
        ("bssfp", "lge")
        if args.dataset == "all"
        else (args.dataset,)
    )

    for dataset_key in selected:
        generate_dataset(dataset_key, clean=args.clean)


if __name__ == "__main__":
    main()
