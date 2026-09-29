"""Module 1: high-dimensional Context-Competitive Matching (CACM).

No anatomical template, ridge classifier, compressed invariant bottleneck or
query labels. Counterclass references are fitted afresh in each episode.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class InvalidSupportEpisode(ValueError):
    """Recoverable training-data condition, never a generic numerical error."""


def validate_masks(fg, bg):
    if not torch.isfinite(fg).all() or not torch.isfinite(bg).all():
        raise FloatingPointError("Nonfinite support annotation")
    if fg.sum() <= 0 or bg.sum() <= 0:
        raise InvalidSupportEpisode("Support needs both classes: fg_mass=%.6g bg_mass=%.6g"
                                    % (float(fg.sum()), float(bg.sum())))


def standardize(x):
    center = x - x.mean((-2, -1), keepdim=True)
    return center / (center.square().mean((-2, -1), keepdim=True) + 1e-5).sqrt()


def logit_parameter(value, low, high):
    p = (float(value) - low) / (high - low)
    if not 0 < p < 1:
        raise ValueError("Initial value must be strictly inside parameter bounds")
    return nn.Parameter(torch.tensor(math.log(p / (1 - p)), dtype=torch.float32))


def balanced_bce(logit, labels):
    loss = F.binary_cross_entropy_with_logits(logit, labels, reduction="none")
    return .5 * ((loss * labels).sum() / labels.sum().clamp_min(1e-6) +
                 (loss * (1 - labels)).sum() / (1 - labels).sum().clamp_min(1e-6))


class CounterClassAnchorCompetition(nn.Module):
    def __init__(self, channels=256, hidden=128, topk=12, opponent_topk=8,
                 max_tokens=1024, chunk_size=256, block_size=4,
                 use_competition=True, initial_competition=.15, revision=1):
        super().__init__()
        if min(channels, hidden, topk, opponent_topk, max_tokens, chunk_size, block_size) < 1:
            raise ValueError("CACM dimensions and counts must be positive")
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden), nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False))
        nn.init.zeros_(self.adapter[-1].weight)
        self.adapter_scale = nn.Parameter(torch.tensor(-2.0))
        self.scale_raw = logit_parameter(10., 4., 20.)
        self.competition_raw = logit_parameter(initial_competition, 0., .5)
        self.use_competition = bool(use_competition)
        self.topk, self.opponent_topk = int(topk), int(opponent_topk)
        self.max_tokens, self.chunk_size = int(max_tokens), int(chunk_size)
        self.block_size = int(block_size)
        self.revision = int(revision)
        if self.revision not in (1, 2):
            raise ValueError("Supported CARE revisions: 1, 2")

    def scale(self):
        return 4. + 16. * self.scale_raw.sigmoid()

    def competition(self):
        return .5 * self.competition_raw.sigmoid() if self.use_competition else self.scale_raw * 0

    def describe(self, features):
        # Retain every backbone channel. This starts as the SIAF semantic
        # descriptor, rather than learning a new low-dimensional space.
        residual = .35 * self.adapter_scale.sigmoid() * self.adapter(standardize(features))
        return F.normalize(standardize(features + residual), dim=1, eps=1e-6)

    def support_banks(self, z, fg, bg):
        """Keep spatially distributed dense tokens, with block-fold provenance.

        A tiny class gets an occupancy-weighted descriptor, not a fake label
        or a forced skip. A singleton descriptor cannot pass the two-fold audit.
        """
        if self.revision >= 2:
            return self._occupancy_banks(z, fg, bg)
        n, c, h, w = z.shape
        tokens = z.permute(0, 2, 3, 1).reshape(-1, c)
        yy = torch.arange(h, device=z.device)[:, None].expand(h, w)
        xx = torch.arange(w, device=z.device)[None, :].expand(h, w)
        folds = ((yy // self.block_size + xx // self.block_size) % 2).flatten().repeat(n)
        banks = []
        valid = (fg + bg).flatten()
        for mask in (bg, fg):
            occupancy = mask.flatten()
            # Require a class-dominant cell, including partially ignored cells.
            fraction = occupancy / valid.clamp_min(1e-6)
            indices = torch.where((fraction > .5) & (occupancy > 0))[0]
            if indices.numel() == 0:
                v = (tokens * occupancy[:, None]).sum(0) / occupancy.sum().clamp_min(1e-12)
                banks.append({"z": F.normalize(v[None], dim=1, eps=1e-6),
                              "fold": folds.new_full((1,), -1)})
                continue
            if indices.numel() > self.max_tokens:
                positions = torch.linspace(0, indices.numel() - 1, self.max_tokens,
                                           device=z.device).long()
                indices = indices[positions]
            banks.append({"z": tokens[indices], "fold": folds[indices]})
        return banks  # background, foreground

    @staticmethod
    def _uniform_indices(count, budget, device):
        if budget <= 0:
            return torch.empty(0, device=device, dtype=torch.long)
        if count <= budget:
            return torch.arange(count, device=device)
        return torch.linspace(0, count - 1, budget, device=device).long()

    def _occupancy_banks(self, z, fg, bg):
        """Dense anchors + class-masked local descriptors for mixed cells.

        All nonzero class occupancy can contribute through regional pooling;
        a thin class is not collapsed into a single global prototype. Regions
        are 2x2 on the native feature grid, not on the input-image grid.
        This cannot recreate detail missing from the encoder representation.
        """
        validate_masks(fg, bg)
        n, c, h, w = z.shape
        tokens = z.permute(0, 2, 3, 1).reshape(-1, c)
        valid = fg + bg
        tile = (min(2, h), min(2, w))
        def pool(x):
            return F.avg_pool2d(x, tile, stride=tile, ceil_mode=True, count_include_pad=False)
        pooled_valid = pool(valid)
        mixed = ((fg > 0) & (bg > 0)).float()
        mixed_regions = pool(mixed).flatten() > 0
        banks = []
        for mask in (bg, fg):
            fraction = mask / valid.clamp_min(1e-6)
            core = torch.where((fraction.flatten() > .5) & (mask.flatten() > 0))[0]
            mass = pool(mask)
            region_features = pool(z * mask) / mass.clamp_min(1e-12)
            # Reserve spatial diversity specifically for partial-volume regions.
            selected = (mass.flatten() > 0) & mixed_regions
            if core.numel() == 0:
                selected = mass.flatten() > 0
            regional = F.normalize(region_features.permute(0, 2, 3, 1).reshape(-1, c)[selected], dim=1, eps=1e-6)
            regional_quality = (mass / pooled_valid.clamp_min(1e-12)).flatten()[selected]
            regional_budget = min(len(regional), max(1, self.max_tokens // 4))
            if core.numel() == 0:
                regional_budget = self.max_tokens
            ri = self._uniform_indices(len(regional), regional_budget, z.device)
            ci = self._uniform_indices(len(core), max(0, self.max_tokens - len(ri)), z.device)
            core = core[ci]
            vectors = torch.cat((tokens[core], regional[ri]))
            quality = torch.cat((fraction.flatten()[core], regional_quality[ri]))
            if not len(vectors):
                raise RuntimeError("Positive support mass produced an empty descriptor bank")
            banks.append({"z": vectors, "quality": quality,
                          "fold": torch.full((len(vectors),), -1, device=z.device, dtype=torch.long)})
        # The revision-2 audit evaluates actual held-out feature cells with
        # soft occupancy labels, NOT these class-conditioned prototype labels.
        banks[0]["audit_support"] = {"z": z, "fg": fg, "bg": bg}
        return banks

    def fit(self, banks):
        """Pair each support key with its hardest counterclass context.

        d(s)=s-eta*mean(top-k opposite keys), without renormalizing d(s).
        This keeps the physical meaning of a difference of cosine evidence.
        """
        own = [b["z"] for b in banks]
        directions = []
        for category in (0, 1):
            x, other = own[category], own[1 - category]
            if not self.use_competition:
                directions.append(x)
                continue
            parts = []
            for chunk in x.split(self.chunk_size):
                similarity = chunk @ other.t()
                indices = similarity.topk(min(self.opponent_topk, other.shape[0]), dim=1).indices
                context = other[indices].mean(1)
                if self.revision >= 2 and "quality" in banks[category]:
                    start = len(parts) * self.chunk_size
                    own_quality = banks[category]["quality"][start:start + len(chunk)]
                    other_quality = banks[1 - category]["quality"][indices].mean(1)
                    reliability = (own_quality * other_quality).clamp_min(0).sqrt()
                    parts.append(chunk - self.competition() * reliability[:, None] * context)
                else:
                    parts.append(chunk - self.competition() * context)
            directions.append(torch.cat(parts))
        return {"raw": own, "competitive": directions}

    def score(self, query, rules, competitive=True):
        """Input [P,C], output foreground-vs-background log-odds [P]."""
        keys = rules["competitive" if competitive else "raw"]
        result = []
        for chunk in query.split(self.chunk_size):
            scores = [(chunk @ bank.t()).topk(min(self.topk, bank.shape[0]), dim=1).values.mean(1)
                      for bank in keys]
            result.append(self.scale() * (scores[1] - scores[0]))
        return torch.cat(result)

    def forward(self, features, fg, bg, shots):
        z = self.describe(features)
        banks = self.support_banks(z[:shots], fg, bg)
        rules = self.fit(banks)
        query = z[shots:].permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        anchor = self.score(query, rules, competitive=False)
        initial = self.score(query, rules) if self.use_competition else anchor
        return z, banks, initial, anchor
