"""Module 2: Support-Audited Regional Adaptation (RAQA).

Spatial cross-fitting checks query-derived updates on held-out support tokens.
It is a support-risk safeguard, NOT a theorem about unseen query Dice.
No target-query labels or optimizer steps are used at inference.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .anchor_competition import balanced_bce, logit_parameter


class RiskAuditedQueryAdaptation(nn.Module):
    def __init__(self, grid=8, topk=3, confidence=.65, min_mass=1.,
                 correction_cap=4., audit_margin=.002, use_adaptation=True,
                 use_audit=True, chunk_size=256, revision=1):
        super().__init__()
        if min(grid, topk, min_mass, correction_cap, chunk_size) <= 0:
            raise ValueError("RAQA sizes and masses must be positive")
        if not .5 < confidence < 1 or audit_margin < 0:
            raise ValueError("Use .5 < confidence < 1 and audit_margin >= 0")
        self.query_scale_raw = logit_parameter(10., 4., 20.)
        self.grid, self.topk, self.chunk_size = int(grid), int(topk), int(chunk_size)
        self.confidence, self.min_mass = float(confidence), float(min_mass)
        self.correction_cap, self.audit_margin = float(correction_cap), float(audit_margin)
        self.use_adaptation, self.use_audit = bool(use_adaptation), bool(use_audit)
        self.revision = int(revision)
        if self.revision not in (1, 2):
            raise ValueError("Supported CARE revisions: 1, 2")
        self.register_buffer("candidates", torch.tensor([0., .25, .5, .75]))

    def query_bank(self, qz, logit):
        """Multiple regional modes; never impose a foreground area prior."""
        h, w = qz.shape[-2:]
        probability = logit.detach().sigmoid().reshape(1, 1, h, w)
        banks = []
        for p in (1 - probability, probability):
            weights = p.square() * (p >= self.confidence).float()
            if float(weights.sum()) < self.min_mass:
                return None
            grid = (min(self.grid, h), min(self.grid, w))
            mass = F.adaptive_avg_pool2d(weights, grid)
            mean = F.adaptive_avg_pool2d(qz * weights, grid) / mass.clamp_min(1e-8)
            vectors = mean.flatten(2)[0].t()[mass.flatten() > 1e-4]
            global_vector = (qz * weights).sum((-2, -1)) / weights.sum().clamp_min(1e-8)
            banks.append(F.normalize(torch.cat((vectors, global_vector)), dim=1, eps=1e-6))
        return banks

    def score(self, x, query_bank):
        scale = 4. + 16. * self.query_scale_raw.sigmoid()
        parts = []
        for chunk in x.split(self.chunk_size):
            score = [(chunk @ b.t()).topk(min(self.topk, len(b)), dim=1).values.mean(1)
                     for b in query_bank]
            parts.append(scale * (score[1] - score[0]))
        return torch.cat(parts)

    def correction(self, baseline, candidate):
        # Strong support evidence stays anchored; weak evidence can change more.
        p = baseline.detach().sigmoid()
        uncertainty = .1 + .9 * 4 * p * (1 - p)
        return self.correction_cap * torch.tanh((candidate - baseline) / self.correction_cap) * uncertainty

    def select_alpha(self, risks):
        """Every fold must improve; alpha=0 always remains admissible."""
        risks = risks.detach()
        safe = ((risks[:, :1] - risks) >= self.audit_margin).all(0)
        safe[0] = True
        mean = risks.mean(0).masked_fill(~safe, float("inf"))
        index = mean.argmin()
        return self.candidates[index], index

    @torch.no_grad()
    def audit(self, qz, banks, matcher):
        if self.revision >= 2 and "audit_support" in banks[0]:
            return self._audit_regions(qz, banks[0]["audit_support"], matcher)
        # Both the pseudo-label construction AND the counterclass references
        # exclude each fold's held-out support keys. Merely auditing predictions
        # from a bank that already contains the held-out labels would be circular.
        q = qz.permute(0, 2, 3, 1).reshape(-1, qz.shape[1])
        risks = []
        for fold in (0, 1):
            train, held, labels = [], [], []
            for category, bank in enumerate(banks):
                keep = bank["fold"] == 1 - fold
                test = bank["fold"] == fold
                if int(keep.sum()) < 2 or int(test.sum()) < 2:
                    return q.new_tensor(0.), {}, "insufficient_support_blocks"
                train.append({"z": bank["z"][keep]})
                held.append(bank["z"][test])
                labels.append(q.new_full((int(test.sum()),), float(category)))
            rules = matcher.fit(train)
            query_bank = self.query_bank(qz, matcher.score(q, rules))
            if query_bank is None:
                return q.new_tensor(0.), {}, "insufficient_fold_query_evidence"
            x, y = torch.cat(held), torch.cat(labels)
            baseline = matcher.score(x, rules)
            delta = self.correction(baseline, self.score(x, query_bank))
            risks.append(torch.stack([balanced_bce(baseline + a * delta, y) for a in self.candidates]))
        risks = torch.stack(risks)
        alpha, index = self.select_alpha(risks)
        info = {"audit_before": risks[:, 0].mean(), "audit_after": risks[:, index].mean(),
                "audit_risks": risks}
        return alpha, info, "accepted" if float(alpha) > 0 else "support_risk_rejected"

    @staticmethod
    def soft_bce(logit, y, weight):
        """Class-balanced BCE from the actual soft support occupancy."""
        fg, bg = weight * y, weight * (1 - y)
        return .5 * ((F.softplus(-logit) * fg).sum() / fg.sum().clamp_min(1e-12) +
                     (F.softplus(logit) * bg).sum() / bg.sum().clamp_min(1e-12))

    @classmethod
    def region_risk(cls, logit, y, weight):
        p = logit.sigmoid()
        def dice(a, b):
            return 1 - (2 * (a * b * weight).sum() + 1e-6) / ((a.square() + b.square()) * weight).sum().add(1e-6)
        overlap = .5 * (dice(p, y) + dice(1 - p, 1 - y))
        return cls.soft_bce(logit, y, weight) + .5 * overlap

    def select_region_alpha(self, risks, boundary_risks):
        risks, boundary_risks = risks.detach(), boundary_risks.detach()
        improves_region = ((risks[:, :1] - risks) >= self.audit_margin).all(0)
        protects_boundary = (boundary_risks <= boundary_risks[:, :1] + 1e-6).all(0)
        safe = improves_region & protects_boundary
        safe[0] = True
        index = risks.mean(0).masked_fill(~safe, float("inf")).argmin()
        return self.candidates[index], index

    @torch.no_grad()
    def _audit_regions(self, qz, support, matcher):
        z, fg, bg = support["z"], support["fg"], support["bg"]
        h, w = z.shape[-2:]
        yy = torch.arange(h, device=z.device)[:, None]
        xx = torch.arange(w, device=z.device)[None, :]
        fold_map = ((yy // matcher.block_size + xx // matcher.block_size) % 2)[None, None]
        valid = fg + bg
        truth = fg / valid.clamp_min(1e-12)
        # Mixed cells and spatial transitions both count as boundary evidence.
        high = F.max_pool2d(truth, 3, stride=1, padding=1)
        low = -F.max_pool2d(-truth, 3, stride=1, padding=1)
        neighborhood_valid = -F.max_pool2d(-valid, 3, stride=1, padding=1)
        boundary = torch.maximum(4 * truth * (1 - truth),
                                 (high - low) * (neighborhood_valid > 0)).clamp(0, 1)
        query = qz.permute(0, 2, 3, 1).reshape(-1, qz.shape[1])
        features = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        risks, edge_risks, query_probabilities = [], [], []
        for fold in (0, 1):
            train = (fold_map != fold).expand_as(valid).float()
            held = (fold_map == fold).expand_as(valid).float()
            for mass in (fg, bg):
                if int(((mass * train) > 0).sum()) < 2 or int(((mass * held) > 0).sum()) < 2:
                    return query.new_tensor(0.), {}, "insufficient_support_blocks"
            # Re-pool with held-out occupancies zeroed. A regional descriptor
            # may never mix held-out labels/features back into its training bank.
            rules = matcher.fit(matcher.support_banks(z, fg * train, bg * train))
            qlogit = matcher.score(query, rules)
            query_probabilities.append(qlogit.sigmoid())
            bank = self.query_bank(qz, qlogit)
            if bank is None:
                return query.new_tensor(0.), {}, "insufficient_fold_query_evidence"
            selected = ((valid * held).flatten() > 0)
            x = features[selected]
            y, weight = truth.flatten()[selected], valid.flatten()[selected]
            edge_weight = weight * boundary.flatten()[selected]
            baseline = matcher.score(x, rules)
            delta = self.correction(baseline, self.score(x, bank))
            candidates = [baseline + alpha * delta for alpha in self.candidates]
            risks.append(torch.stack([self.region_risk(v, y, weight) for v in candidates]))
            edge_risks.append(torch.stack([self.soft_bce(v, y, edge_weight) for v in candidates]))
        risks, edge_risks = torch.stack(risks), torch.stack(edge_risks)
        alpha, index = self.select_region_alpha(risks, edge_risks)
        p0, p1 = query_probabilities
        agreement = ((p0 >= .5) == (p1 >= .5)).float() * (1 - (p0 - p1).abs())
        # This additional query-side safeguard is not an estimate of query DSC.
        info = {"audit_before": risks[:, 0].mean(), "audit_after": risks[:, index].mean(),
                "audit_boundary_before": edge_risks[:, 0].mean(),
                "audit_boundary_after": edge_risks[:, index].mean(),
                "audit_risks": risks, "audit_boundary_risks": edge_risks,
                "query_agreement": agreement}
        return alpha, info, "accepted" if float(alpha) > 0 else "support_risk_rejected"

    def forward(self, qz, banks, initial, matcher, training_objectives=True):
        zero = initial.sum() * 0 + self.query_scale_raw * 0
        info = {"alpha": zero.detach(), "reverse_loss": zero,
                "audit_before": zero.detach(), "audit_after": zero.detach(),
                "reason": "disabled"}
        if not self.use_adaptation:
            return initial, info
        bank = self.query_bank(qz, initial)
        if bank is None:
            info["reason"] = "insufficient_query_evidence"
            return initial, info
        if self.use_audit:
            alpha, audit_info, reason = self.audit(qz.detach(), banks, matcher)
            info.update(audit_info)
        else:
            alpha, reason = self.candidates[-1], "unaudited_ablation"
        q = qz.permute(0, 2, 3, 1).reshape(-1, qz.shape[1])
        delta = self.correction(initial, self.score(q, bank))
        if "query_agreement" in info:
            delta = delta * info["query_agreement"].detach()
        final = initial + alpha.detach() * delta
        if training_objectives:
            # Trains useful query evidence even on episodes where the detached
            # audit rejects adaptation; does not train the acceptance rule.
            if self.revision >= 2 and "audit_support" in banks[0]:
                data = banks[0]["audit_support"]
                support = data["z"].permute(0, 2, 3, 1).reshape(-1, qz.shape[1])
                weight = (data["fg"] + data["bg"]).flatten()
                labels = data["fg"].flatten() / weight.clamp_min(1e-12)
                info["reverse_loss"] = self.region_risk(self.score(support, bank), labels, weight)
            else:
                support = torch.cat([b["z"] for b in banks])
                labels = torch.cat([support.new_full((len(b["z"]),), float(i)) for i, b in enumerate(banks)])
                info["reverse_loss"] = balanced_bce(self.score(support, bank), labels)
        info.update({"alpha": alpha.detach(), "reason": reason,
                     "correction_abs": (alpha * delta).abs().mean().detach()})
        return final, info
