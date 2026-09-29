"""Explicit-protocol NIfTI evaluation, retaining the supplied legacy metric.

same_scan_fg: original protocol, median positive support slice, other positive
slices in the same scan. same_scan_all: all other slices, including absence.
cross_scan_all: a user-fixed separate support scan, all query scans/slices.
Scan separation is not necessarily patient separation; verify dataset IDs.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("TORCH_HOME", "./pretrained_model")
from config_care_test import ex
from care_defaults import MODEL, merge_config
from models.care import CARENet
from care_compat import (load_checkpoint, run_cli, model_config_from_checkpoint,
                         model_state_from_checkpoint)
from care_reporting import label_names, organ_statistics, print_organ_summary, write_organ_csv


def checkpoint_model_config(checkpoint, overrides):
    saved = model_config_from_checkpoint(checkpoint)
    config = merge_config(MODEL, saved)
    # Absence means the original revision, not the new training default.
    config["care"]["revision"] = saved["care"].get("revision", 1)
    config = merge_config(config, overrides)
    config["use_coco_init"] = False
    return config


def scan_id(path):
    name = Path(path).name
    if name.startswith("image_"):
        name = name[len("image_"):]
    if name.endswith(".nii.gz"):
        name = name[:-len(".nii.gz")]
    return name


def label_path(path):
    p = Path(path)
    if not p.name.startswith("image_"):
        raise ValueError("Expected image_<id>.nii.gz: " + str(path))
    return p.with_name("label_" + p.name[len("image_"):])


def resize_volume(volume, target, is_label=False):
    tensor = torch.from_numpy(volume.astype(np.float32))[:, None]
    if is_label:
        tensor = F.interpolate(tensor, size=(target, target), mode="nearest")
    else:
        tensor = F.interpolate(tensor, size=(target, target), mode="bilinear", align_corners=False)
    return tensor[:, 0].numpy()


def load_scan(path, target_size):
    import SimpleITK as sitk
    lp = label_path(path)
    if not lp.is_file():
        raise FileNotFoundError("Missing evaluation annotation: " + str(lp))
    image_obj, label_obj = sitk.ReadImage(str(path)), sitk.ReadImage(str(lp))
    image, labels = sitk.GetArrayFromImage(image_obj), sitk.GetArrayFromImage(label_obj)
    if image.ndim != 3 or labels.shape != image.shape:
        raise ValueError("Image/label volumes must have identical [D,H,W] shapes")
    for attr in ("GetSpacing", "GetOrigin", "GetDirection"):
        if not np.allclose(getattr(image_obj, attr)(), getattr(label_obj, attr)()):
            raise ValueError("Image/label physical geometry differs: " + str(path))
    image = resize_volume(image, target_size)
    labels = resize_volume(labels, target_size, True).astype(np.int32)
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite image: " + str(path))
    image = (image - image.mean()) / (image.std() + 1e-8)
    return image, labels


def dice_score(pred, truth):
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    denominator = pred.sum() + truth.sum()
    return 1. if denominator == 0 else float(2 * (pred & truth).sum() / denominator)


def image_tensor(image, device):
    return torch.from_numpy(image.copy()).float()[None, None].to(device).repeat(1, 3, 1, 1)


def make_support(image, labels, category, device, n_shots=1):
    y = labels == category
    slices = np.flatnonzero(y.sum((1, 2)) > 0)
    if len(slices) < n_shots:
        return None
    positions = np.floor((np.arange(n_shots) + .5) * len(slices) / n_shots).astype(int)
    zs = [int(slices[p]) for p in positions]
    masks = [torch.from_numpy(y[z].astype(np.float32))[None].to(device) for z in zs]
    return [[image_tensor(image[z], device) for z in zs]], [masks], [[1 - m for m in masks]], zs


@torch.no_grad()
def predict_episode(model, support, fg, bg, query, use_flip=False):
    logits, _, _ = model(support, fg, bg, query, isval=True)
    if use_flip:
        flip = lambda nested: [[v.flip(-1) for v in way] for way in nested]
        other, _, _ = model(flip(support), flip(fg), flip(bg), [query[0].flip(-1)], isval=True)
        logits = (logits + other.flip(-1)) / 2
    return logits


@torch.no_grad()
def evaluate_domain(model, domain, cfg, device, protocol, support_id, use_flip, n_shots=1):
    names, name_source = label_names(domain, cfg)
    for category in cfg["test_label"]:
        if int(category) not in names:
            raise ValueError("Missing organ name for label %s" % category)
    print("  organ mapping : " + name_source)
    pattern = os.path.join(cfg["data_dir"], "image*.nii.gz")
    files = sorted(glob.glob(pattern), key=lambda p: (0, int(scan_id(p))) if scan_id(p).isdigit() else (1, scan_id(p)))
    if not files:
        raise FileNotFoundError("No image volumes: " + pattern)
    print("  input size    : %dx%d" % (cfg["target_size"], cfg["target_size"]))
    print("  protocol      : " + protocol)
    cross_support, cross_path = {}, None
    if protocol == "cross_scan_all":
        matches = [p for p in files if scan_id(p) == support_id]
        if len(matches) != 1:
            raise ValueError("cross_scan_all requires one valid support_scan_id")
        cross_path = matches[0]
        image, labels = load_scan(cross_path, cfg["target_size"])
        for category in cfg["test_label"]:
            support = make_support(image, labels, category, device, n_shots)
            if support is None:
                raise ValueError("Fixed support scan has too few positive slices for category %s" % category)
            cross_support[category] = support
    records, skips = [], []
    for path in files:
        if path == cross_path:
            continue
        image, labels = load_scan(path, cfg["target_size"])
        for category in cfg["test_label"]:
            truth = labels == category
            support = cross_support[category] if cross_path else make_support(image, labels, category, device, n_shots)
            if support is None:
                skips.append({"scan": scan_id(path), "class": category, "reason": "insufficient_positive_support_slices"})
                continue
            si, fg, bg, support_z = support
            if protocol == "same_scan_fg":
                indices = [int(z) for z in np.flatnonzero(truth.sum((1, 2)) > 0) if z not in support_z]
            else:
                indices = [z for z in range(len(image)) if cross_path or z not in support_z]
            if not indices:
                skips.append({"scan": scan_id(path), "class": category, "reason": "no_query_slices"})
                continue
            scores, positive_scores, empty_fp, empty_area = [], [], [], []
            intersection, predicted_count, gt_count = 0, 0, 0
            alphas, audit_reasons = [], {}
            for z in indices:
                logits = predict_episode(model, si, fg, bg, [image_tensor(image[z], device)], use_flip)
                alphas.append(float(model._last_care["alpha"]))
                reason = model._last_care["reason"]
                audit_reasons[reason] = audit_reasons.get(reason, 0) + 1
                pred = logits.argmax(1)[0].cpu().numpy().astype(bool)
                gt = truth[z]
                dsc = dice_score(pred, gt)
                scores.append(dsc)
                if gt.any():
                    positive_scores.append(dsc)
                else:
                    empty_fp.append(float(pred.any()))
                    empty_area.append(float(pred.mean()))
                intersection += int((pred & gt).sum())
                predicted_count += int(pred.sum())
                gt_count += int(gt.sum())
            denom = predicted_count + gt_count
            record = {"scan": scan_id(path), "class": int(category), "organ": names[int(category)],
                      "support_scan": scan_id(cross_path or path),
                      "support_slice": support_z[0] if n_shots == 1 else None, "support_slices": support_z,
                      "adaptation_acceptance": float(np.mean(np.array(alphas) > 0)),
                      "adaptation_mean_alpha": float(np.mean(alphas)), "audit_reasons": audit_reasons,
                      "query_slices": indices, "slice_mean_dsc": float(np.mean(scores)),
                      "positive_slice_mean_dsc": float(np.mean(positive_scores)) if positive_scores else None,
                      "evaluated_voxel_dsc": float(2 * intersection / denom) if denom else 1.,
                      "empty_count": len(empty_fp),
                      "empty_false_positive_slices": int(sum(empty_fp)),
                      "empty_mean_predicted_area": float(np.mean(empty_area)) if empty_area else None}
            records.append(record)
            print("scan=%s class=%s organ=%s slice_DSC=%.4f voxel_DSC=%.4f empty=%d" %
                  (record["scan"], category, names[int(category)], record["slice_mean_dsc"], record["evaluated_voxel_dsc"], len(empty_fp)), flush=True)
    if not records:
        raise RuntimeError("No valid evaluated episodes in " + domain)
    values = [r["slice_mean_dsc"] for r in records]
    means = {str(c): float(np.mean([r["slice_mean_dsc"] for r in records if r["class"] == c]))
             for c in cfg["test_label"] if any(r["class"] == c for r in records)}
    empty_count = sum(r["empty_count"] for r in records)
    fp_count = sum(r["empty_false_positive_slices"] for r in records)
    query_count = sum(len(r["query_slices"]) for r in records)
    acceptance = sum(r["adaptation_acceptance"] * len(r["query_slices"]) for r in records) / query_count
    mean_alpha = sum(r["adaptation_mean_alpha"] * len(r["query_slices"]) for r in records) / query_count
    result = {"domain": domain, "protocol": protocol, "n_shots": n_shots,
            "adaptation_acceptance": acceptance, "adaptation_mean_alpha": mean_alpha,
            "audit_diagnostic_view": "flipped" if use_flip else "original",
            "legacy_scan_class_mean_dsc": float(np.mean(values)),
            "class_macro_mean_dsc": float(np.mean(list(means.values()))),
            "class_means": means, "std": float(np.std(values)), "N": len(values),
            "empty_query_slices": empty_count,
            "empty_slice_false_positive_rate": fp_count / empty_count if empty_count else None,
            "records": records, "skips": skips,
            "organ_name_source": name_source,
            "organ_statistics": organ_statistics(records, cfg["test_label"], names)}
    print_organ_summary(result)
    print("[ADAPTATION] accepted_fraction=%.4f mean_alpha=%.4f" % (acceptance, mean_alpha))
    return result


@ex.main
def main(_run, _config, _log, model):
    # Explicitly capture this config subtree. Reading it only via _config
    # does not mark newly supplied model.* CLI keys as used in Sacred.
    del _log
    protocol = _config["protocol"]
    support_id = str(_config["support_scan_id"]).strip()
    if _config["n_shots"] < 1:
        raise ValueError("n_shots must be positive")
    if protocol not in {"same_scan_fg", "same_scan_all", "cross_scan_all"}:
        raise ValueError("Unknown evaluation protocol: " + protocol)
    if protocol == "cross_scan_all" and not support_id:
        raise ValueError("Set support_scan_id for cross_scan_all before evaluation")
    torch.set_num_threads(1)
    torch.manual_seed(_config["seed"])
    np.random.seed(_config["seed"])
    device = torch.device("cuda:%d" % _config["gpu_id"] if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(_config["reload_model_path"])
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if not all(key in checkpoint for key in ("model_config", "state_dict")):
        raise ValueError("Expected a CARE-Net checkpoint with architecture metadata")
    if _config["expected_source"] and checkpoint.get("train_config", {}).get("dataset") != _config["expected_source"]:
        raise ValueError("Checkpoint training source differs from expected_source")
    config = checkpoint_model_config(checkpoint, model)
    checkpoint_revision = model_config_from_checkpoint(checkpoint)["care"].get("revision", 1)
    print("[CARE_REVISION] checkpoint=%d inference=%d" % (checkpoint_revision, config["care"]["revision"]))
    model = CARENet(cfg=config).to(device)
    model.load_state_dict(model_state_from_checkpoint(checkpoint), strict=True)
    model.eval()
    summary = {"checkpoint": str(checkpoint_path.resolve()), "step": checkpoint["step"],
               "checkpoint_revision": checkpoint_revision, "inference_revision": config["care"]["revision"],
               "training_source": checkpoint.get("train_config", {}).get("dataset"),
               "seed": _config["seed"], "n_shots": _config["n_shots"],
               "model_config": config, "use_flip_tta": _config["use_horizontal_flip_tta"], "domains": {}}
    for domain in _config["eval_domains"]:
        result = evaluate_domain(model, domain, _config["path"][domain], device,
                                 protocol, support_id, _config["use_horizontal_flip_tta"], _config["n_shots"])
        summary["domains"][domain] = result
        _run.log_scalar(domain + "/mean_dsc", result["legacy_scan_class_mean_dsc"])
        for row in result["organ_statistics"].values():
            if row["N"]:
                _run.log_scalar("%s/label_%d/mean_dsc" % (domain, row["label"]), row["mean_dsc"])
    directory = Path(_config["result_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    # A new timestamped result avoids overwriting another seed/run/protocol.
    from datetime import datetime, timezone
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / (protocol + "_" + suffix + ".json")
    for result in summary["domains"].values():
        result["source_result"], result["checkpoint"] = str(path.resolve()), str(checkpoint_path.resolve())
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    write_organ_csv(path.with_suffix(".csv"), summary["domains"])
    print("[RESULT_JSON] " + str(path.resolve()))
    print("[ORGAN_CSV] " + str(path.with_suffix(".csv").resolve()))
    return str(path)


if __name__ == "__main__":
    run_cli(ex)
