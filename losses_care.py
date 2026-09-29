"""Source-episode losses; query labels enter only here, never the model."""
import torch
from torch.nn import functional as F


def resolve_class_weights(cfg, device):
    explicit = cfg.get("ce_class_weights")
    if explicit is not None:
        value = explicit
    elif not cfg.get("use_wce", False):
        value = [1., 1.]
    elif cfg["dataset"] in ("Prostate_NCI", "Prostate_UCLH"):
        value = [.05, 1.]  # exact convention in uploaded train_siaf_ra.py
    else:
        try:
            from util.utils import compose_wt_simple
        except (ModuleNotFoundError, ImportError) as error:
            raise ImportError("Weighted CE needs your original util.utils.compose_wt_simple. "
                              "Retain that file or set explicit ce_class_weights=[BG,FG]; "
                              "use_wce=False explicitly selects unweighted CE.") from error
        value = compose_wt_simple(True, cfg["dataset"])
    weight = torch.as_tensor(value, device=device, dtype=torch.float32).detach()
    if weight.shape != (2,) or not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("CE weights must be two finite positive values [BG,FG]")
    return weight


def segmentation_loss(logits, target, ignore=255, dice_weight=.5, class_weights=None):
    if target.ndim == 2:
        target = target[None]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("Target must be [B,H,W]")
    target = target.long()
    valid = target != ignore
    if not valid.any():
        return logits.sum() * 0
    if not torch.all((target[valid] == 0) | (target[valid] == 1)):
        raise ValueError("Query labels must be binary or ignored")
    weight = None if class_weights is None else torch.as_tensor(class_weights, device=logits.device, dtype=logits.dtype)
    if weight is not None and (weight.shape != (2,) or not torch.isfinite(weight).all() or (weight <= 0).any()):
        raise ValueError("CE weights must be positive [BG,FG]")
    ce = F.cross_entropy(logits, target, ignore_index=ignore, weight=weight)
    p, y = logits.softmax(1)[:, 1] * valid, (target == 1).float() * valid
    ps, ys = p.sum((-2, -1)), y.sum((-2, -1))
    dice = 1 - (2 * (p * y).sum((-2, -1)) + 1e-6) / (ps + ys + 1e-6)
    empty = ps / valid.sum((-2, -1)).clamp_min(1)
    return ce + dice_weight * torch.where(ys > 0, dice, empty).mean()


def compute_losses(prediction, anchor, aux, target, cfg):
    ignore, dw = int(cfg.get("ignore_label", 255)), float(cfg.get("lambda_dice", .5))
    weight = cfg.get("ce_class_weights")
    terms = {"segmentation": segmentation_loss(prediction, target, ignore, dw, weight),
             "anchor": segmentation_loss(anchor, target, ignore, dw, weight),
             "competitive": segmentation_loss(aux["competitive_logits"], target, ignore, dw, weight),
             "reverse": aux["reverse_loss"]}
    total = terms["segmentation"] + float(cfg.get("lambda_anchor", .5)) * terms["anchor"]
    total = total + float(cfg.get("lambda_competitive", .25)) * terms["competitive"]
    total = total + float(cfg.get("lambda_reverse", .1)) * terms["reverse"]
    terms["total"] = total
    return total, terms
