"""CARE-Net training on the original TrainDataset episodic API."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

os.environ.setdefault("TORCH_HOME", "./pretrained_model")
from config_care_train import ex
from dataloaders.datasets import TrainDataset
from models.care import CARENet
from models.anchor_competition import InvalidSupportEpisode
from losses_care import compute_losses, resolve_class_weights
from care_compat import run_cli

_DATASET_NAME_MAP = {
    "CHAOST2": "ABDOMEN_MR",
    "SABS": "ABDOMEN_CT",
    "CARDIAC_bssFP": "CARDIAC_bssFP",
    "CARDIAC_LGE": "CARDIAC_LGE",
    "Prostate_NCI": "Prostate_NCI",
    "Prostate_UCLH": "Prostate_UCLH",
}

_TRAIN_NORMALIZED_SUBDIRS = {
    "ABDOMEN_MR": "chaos_MR_T2_normalized",
    "ABDOMEN_CT": "sabs_CT_normalized",
    "CARDIAC_bssFP": "cmr_bssFP_normalized",
    "CARDIAC_LGE": "cmr_LGE_normalized",
    "Prostate_NCI": "NCI_normalized",
    "Prostate_UCLH": "UCLH_normalized",
}


def _adnet_dataset_and_dir(config_dataset, config_data_dir):
    adnet_name = _DATASET_NAME_MAP.get(config_dataset, config_dataset)
    data_dir = os.path.normpath(config_data_dir)
    if adnet_name not in _TRAIN_NORMALIZED_SUBDIRS:
        raise ValueError("Unsupported training dataset: %s" % config_dataset)
    normalized_name = _TRAIN_NORMALIZED_SUBDIRS[adnet_name]
    if os.path.basename(data_dir) == normalized_name:
        data_root = os.path.dirname(data_dir)
    elif os.path.isdir(os.path.join(data_dir, normalized_name)):
        data_root = data_dir
    else:
        data_root = data_dir
    return adnet_name, data_root



def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_optimizer(model, cfg):
    backbone, head = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (backbone if name.startswith("encoder.") else head).append(p)
    lr = float(cfg["lr"])
    return torch.optim.SGD([
        {"params": backbone, "lr": lr * float(cfg["backbone_lr_mult"]), "group_name": "backbone"},
        {"params": head, "lr": lr, "group_name": "care"}],
        momentum=float(cfg["momentum"]), weight_decay=float(cfg["weight_decay"]))


def unpack_episode(batch, device):
    images = batch["support_images"][0]
    masks = batch["support_fg_labels"][0]
    if len(images) != len(masks):
        raise ValueError("Loader support image/mask count mismatch")
    # Preserve the original medical pipeline: first channel -> repeated gray.
    support_images = [[x[:, :1].float().to(device).repeat(1, 3, 1, 1) for x in images]]
    foreground, background = [], []
    for mask in masks:
        mask = mask.float().to(device)
        if mask.ndim == 2:
            mask = mask[None]
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        valid = (mask == 0) | (mask == 1)
        foreground.append(torch.where(valid, mask, torch.zeros_like(mask)))
        background.append(torch.where(valid, 1 - mask, torch.zeros_like(mask)))
    query = batch["query_images"][0][:, :1].float().to(device).repeat(1, 3, 1, 1)
    target = batch["query_labels"][0].long().to(device)
    if target.ndim == 2:
        target = target[None]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    return support_images, [foreground], [background], [query], target


def save_checkpoint(path, model, optimizer, scheduler, step, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": "CARE_NET_v1", "state_dict": model.state_dict(),
                "model_config": dict(config["model"]), "train_config": dict(config),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step}, temporary)
    os.replace(temporary, path)


@ex.main
def main(_run, _config, _log):
    if _config["batch_size"] != 1 or _config["task"]["n_ways"] != 1 or _config["task"]["n_queries"] != 1:
        raise ValueError("Use batch_size=1, one way and one query")
    if _config["n_steps"] <= 0 or _config["max_iters_per_load"] <= 0:
        raise ValueError("Training step counts must be positive")
    if not torch.cuda.is_available():
        _log.info("CUDA unavailable; using CPU (full training will be slow).")
    device = torch.device("cuda:%d" % _config["gpu_id"] if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    seed_all(_config["seed"])
    weights = resolve_class_weights(_config, device)
    effective_config = dict(_config)
    effective_config["ce_class_weights"] = weights.cpu().tolist()
    print("[CE_WEIGHTS] background=%.6g foreground=%.6g" % tuple(effective_config["ce_class_weights"]), flush=True)
    model = CARENet(cfg=_config["model"]).to(device).train()
    print("CARE-Net model config:", json.dumps(dict(_config["model"]), sort_keys=True))
    name, root = _adnet_dataset_and_dir(_config["dataset"], _config["path"][_config["dataset"]]["data_dir"])
    if not os.path.isdir(root):
        raise FileNotFoundError("Training data root does not exist: " + root)
    data_cfg = {"data_dir": root, "dataset": name,
                "n_shot": _config["task"]["n_shots"], "n_way": 1, "n_query": 1,
                "n_sv": _config["n_sv"], "max_iter": _config["max_iters_per_load"],
                "min_size": _config["min_size"], "test_label": _config["test_label"],
                "exclude_label": _config["exclude_label"], "use_gt": _config["use_gt"]}
    dataset = TrainDataset(data_cfg)
    generator = torch.Generator().manual_seed(_config["seed"])
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        num_workers=_config["num_workers"], pin_memory=device.type == "cuda",
                        worker_init_fn=seed_worker, generator=generator,
                        persistent_workers=False, drop_last=True)
    if len(loader) == 0:
        raise RuntimeError("No episodes available in TrainDataset")
    optimizer = build_optimizer(model, _config["optim"])
    scheduler = StepLR(optimizer, step_size=_config["lr_step_every"], gamma=_config["lr_step_gamma"])
    observer_dirs = [o.dir for o in _run.observers if getattr(o, "dir", None)]
    run_dir = Path(observer_dirs[0] if observer_dirs else "./runs/CARE_NET_manual")
    snapshot = run_dir / "snapshots"
    snapshot.mkdir(parents=True, exist_ok=True)
    step, window_count, sums = 0, 0, {}
    attempted, skipped, invalid_streak = 0, 0, 0
    max_invalid_streak = 100
    while step < _config["n_steps"]:
        for batch in loader:
            attempted += 1
            support, fg, bg, query, target = unpack_episode(batch, device)
            optimizer.zero_grad(set_to_none=True)
            try:
                prediction, unary, aux = model(support, fg, bg, query)
            except InvalidSupportEpisode as error:
                skipped += 1
                invalid_streak += 1
                _run.info["support_episode_sampling"] = {
                    "attempted": attempted, "skipped": skipped,
                    "successful_updates": step, "consecutive_invalid": invalid_streak}
                if skipped <= 5 or skipped % 50 == 0:
                    raw_values = [m.detach().unique().cpu().tolist()[:12]
                                  for m in batch["support_fg_labels"][0]]
                    print("[INVALID_SUPPORT] attempt=%d successful_step=%d "
                          "skipped=%d streak=%d: %s raw_mask_values=%s"
                          % (attempted, step, skipped, invalid_streak, error, raw_values),
                          flush=True)
                if invalid_streak >= max_invalid_streak:
                    raise RuntimeError(
                        "100 consecutive invalid support episodes. Check TrainDataset "
                        "label encoding, support crops/augmentation and class sampling; "
                        "see [INVALID_SUPPORT] diagnostics. Last error: %s" % error
                    ) from error
                # Only this known data condition is recoverable. Do not run
                # backward/optimizer/scheduler or increment step. Fetch the
                # next episode; a valid query may still have no foreground.
                continue
            invalid_streak = 0
            total, terms = compute_losses(prediction, unary, aux, target, effective_config)
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite loss before optimizer step %d" % (step + 1))
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), _config["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            _run.info["support_episode_sampling"] = {
                "attempted": attempted, "skipped": skipped,
                "successful_updates": step, "consecutive_invalid": 0}
            window_count += 1
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.) + float(value.detach())
            alpha = float(model._last_care["alpha"])
            sums["accept_fraction"] = sums.get("accept_fraction", 0.) + float(alpha > 0)
            sums["mean_alpha"] = sums.get("mean_alpha", 0.) + alpha
            if step % _config["print_interval"] == 0 or step == _config["n_steps"]:
                averages = {k: v / window_count for k, v in sums.items()}
                for key, value in averages.items():
                    _run.log_scalar(key, value, step)
                for key in ("competition", "alpha", "audit_before", "audit_after"):
                    _run.log_scalar(key, float(model._last_care[key]), step)
                _run.log_scalar("invalid_support_skipped", skipped, step)
                _run.log_scalar("invalid_support_fraction", skipped / float(attempted), step)
                print("step %d/%d %s lr=%.3e skipped=%d last_alpha=%.2f last_audit=%s" %
                      (step, _config["n_steps"], " ".join("%s=%.4f" % kv for kv in averages.items()),
                       optimizer.param_groups[-1]["lr"], skipped,
                       float(model._last_care["alpha"]), model._last_care["reason"]), flush=True)
                sums, window_count = {}, 0
            if step == _config["n_steps"]:
                final = snapshot / ("%d.pth" % step)
                save_checkpoint(final, model, optimizer, scheduler, step, effective_config)
                print("[FINAL_CHECKPOINT] " + str(final.resolve()), flush=True)
                return str(final)
            if _config["checkpoint_interval"] > 0 and step % _config["checkpoint_interval"] == 0:
                save_checkpoint(snapshot / "latest.pth", model, optimizer, scheduler, step, effective_config)
        # Reload between iterator lifetimes, so worker dataset copies refresh.
        if hasattr(dataset, "reload_buffer"):
            dataset.reload_buffer()


if __name__ == "__main__":
    run_cli(ex)
