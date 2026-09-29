"""CARE-Net: shared encoder + exactly two core episodic modules (CACM, RAQA)."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .anchor_competition import CounterClassAnchorCompetition, validate_masks
from .query_adaptation import RiskAuditedQueryAdaptation
from care_compat import (load_checkpoint, model_config_from_checkpoint,
                         model_state_from_checkpoint)


class CARENet(nn.Module):
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None,
                 sdf_criterion=None, encoder=None):
        super().__init__()
        del in_channels, sdf_criterion
        self.config = dict(cfg or {})
        if "care" not in self.config and any(
                isinstance(value, dict) and "feat_dim" in value
                for value in self.config.values()):
            self.config = model_config_from_checkpoint({"model_config": self.config})
            self.config["care"].setdefault("revision", 1)
        c = self.config.get("care", {})
        self.revision = int(c.get("revision", 2))
        if encoder is None:
            from .backbone.torchvision_backbones import TVDeeplabRes50Encoder, TVDeeplabRes101Encoder
            names = {"resnet50": TVDeeplabRes50Encoder, "dlfcn_res50": TVDeeplabRes50Encoder,
                     "resnet101": TVDeeplabRes101Encoder, "dlfcn_res101": TVDeeplabRes101Encoder}
            name = self.config.get("which_model", "dlfcn_res101")
            if name not in names:
                raise ValueError("Unsupported backbone: " + name)
            encoder = names[name](bool(self.config.get("use_coco_init", True)) and not pretrained_path)
        self.encoder = encoder
        self.freeze_bn = bool(c.get("freeze_backbone_bn", True))
        self.max_feature_size = int(c.get("max_feature_size", 64))
        if self.max_feature_size < 4:
            raise ValueError("max_feature_size must be >= 4")
        self.cacm = CounterClassAnchorCompetition(
            channels=int(c.get("feat_dim", 256)), hidden=int(c.get("adapter_hidden", 128)),
            topk=int(c.get("topk", 12)), opponent_topk=int(c.get("opponent_topk", 8)),
            max_tokens=int(c.get("max_support_tokens", 1024)), chunk_size=int(c.get("chunk_size", 256)),
            block_size=int(c.get("audit_block_size", 4)),
            use_competition=bool(c.get("use_competition", True)),
            initial_competition=float(c.get("initial_competition", .15)), revision=self.revision)
        self.raqa = RiskAuditedQueryAdaptation(
            grid=int(c.get("query_grid", 8)), topk=int(c.get("query_topk", 3)),
            confidence=float(c.get("query_confidence", .65)), min_mass=float(c.get("query_min_mass", 1.)),
            correction_cap=float(c.get("correction_cap", 4.)), audit_margin=float(c.get("audit_margin", .002)),
            use_adaptation=bool(c.get("use_adaptation", True)), use_audit=bool(c.get("use_audit", True)),
            chunk_size=int(c.get("chunk_size", 256)), revision=self.revision)
        self._last_care = {}
        if pretrained_path:
            checkpoint = load_checkpoint(Path(pretrained_path), map_location="cpu")
            self.load_state_dict(model_state_from_checkpoint(checkpoint), strict=True)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    @staticmethod
    def stack_masks(nested):
        result = []
        for mask in nested[0]:
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            if mask.ndim != 4 or tuple(mask.shape[:2]) != (1, 1):
                raise ValueError("Support masks must be [1,H,W] or [1,1,H,W]")
            result.append(mask.float())
        return torch.cat(result)

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs,
                isval=False, val_wsize=None, sdf_gt=None, query_gt=None):
        del val_wsize, sdf_gt, query_gt  # Query annotation never enters inference.
        if len(supp_imgs) != 1 or len(qry_imgs) != 1 or len(fore_mask) != 1 or len(back_mask) != 1:
            raise ValueError("Use one-way, one-query episodes")
        shots = len(supp_imgs[0])
        if shots < 1 or len(fore_mask[0]) != shots or len(back_mask[0]) != shots:
            raise ValueError("Support shot counts differ")
        images_list = list(supp_imgs[0]) + list(qry_imgs)
        if any(x.ndim != 4 or x.shape[0] != 1 or x.shape[1:] != images_list[0].shape[1:] for x in images_list):
            raise ValueError("Images must be [1,C,H,W] with identical sizes")
        images = torch.cat(images_list)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        if images.shape[1] != 3 or not torch.isfinite(images).all():
            raise ValueError("Images must have 1 or 3 channels and finite values")
        size = images.shape[-2:]
        fg, bg = self.stack_masks(fore_mask), self.stack_masks(back_mask)
        if fg.shape != bg.shape or fg.shape[-2:] != size:
            raise ValueError("Support image/mask shape mismatch")
        if not torch.isfinite(fg).all() or not torch.isfinite(bg).all():
            raise FloatingPointError("Nonfinite support masks")
        valid = ((fg >= 0) & (fg <= 1) & (bg >= 0) & (bg <= 1) & ((fg + bg) <= 1.00001)).float()
        fg, bg = fg.clamp(0, 1) * valid, bg.clamp(0, 1) * valid
        validate_masks(fg, bg)  # BEFORE encoder; typed invalid-data skip only.
        features = self.encoder(images, low_level=False)
        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            raise TypeError("Project encoder must return [N,C,H,W]")
        if max(features.shape[-2:]) > self.max_feature_size:
            ratio = self.max_feature_size / max(features.shape[-2:])
            fs = tuple(max(2, round(v * ratio)) for v in features.shape[-2:])
            features = F.interpolate(features, size=fs, mode="bilinear", align_corners=False)
        fs = features.shape[-2:]
        fg, bg = F.interpolate(fg, size=fs, mode="area"), F.interpolate(bg, size=fs, mode="area")
        z, banks, initial, anchor = self.cacm(features.float(), fg, bg, shots)
        final, info = self.raqa(z[shots:], banks, initial, self.cacm,
                                training_objectives=self.training and not isval)
        def output(logit):
            logit = logit.reshape(1, 1, *fs)
            return F.interpolate(torch.cat((-.5 * logit, .5 * logit), dim=1),
                                 size=size, mode="bilinear", align_corners=False)
        self._last_care = {key: value.detach() if torch.is_tensor(value) else value
                           for key, value in info.items() if key != "reverse_loss"}
        self._last_care["competition"] = self.cacm.competition().detach()
        aux = {"competitive_logits": output(initial), "reverse_loss": info["reverse_loss"]}
        return output(final), output(anchor), aux


# Preserve the original project constructor import when needed.
FewShotSeg = CARENet
