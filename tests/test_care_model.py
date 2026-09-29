"""Synthetic engineering checks, not a claim of clinical performance."""
import ast
import copy
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.care import CARENet
from models.anchor_competition import CounterClassAnchorCompetition, InvalidSupportEpisode
from models.query_adaptation import RiskAuditedQueryAdaptation
from losses_care import compute_losses, segmentation_loss
from care_defaults import MODEL
from care_reporting import organ_statistics
from compare_care_results import compare_pair

torch.set_num_threads(1)


class ToyEncoder(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x, low_level=False):
        return self.bn(self.conv(F.avg_pool2d(x, 8, ceil_mode=True)))


def make_model(**overrides):
    cfg = copy.deepcopy(MODEL)
    cfg["care"].update(feat_dim=16, adapter_hidden=32, audit_block_size=2)
    cfg["care"].update(overrides)
    torch.manual_seed(9)
    return CARENet(cfg=cfg, encoder=ToyEncoder(cfg["care"]["feat_dim"]))


def episode(size=96, shots=1):
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(1, 3, size, size, generator=generator)
    y = torch.zeros(1, size, size)
    y[:, size // 4:3 * size // 4, size // 4:3 * size // 4] = 1
    x = x + 3 * y[:, None]
    return [[x.clone() for _ in range(shots)]], [[y.clone() for _ in range(shots)]], \
           [[1 - y for _ in range(shots)]], [x + .05 * torch.randn(x.shape, generator=generator)]


class ModelTest(unittest.TestCase):
    def test_exactly_two_core_modules(self):
        self.assertEqual(set(dict(make_model().named_children())), {"encoder", "cacm", "raqa"})

    def test_shapes_full_256_channel_head_and_backward(self):
        for size in (192, 256, 257):
            model = make_model(feat_dim=256, adapter_hidden=128).train()
            args = episode(size)
            pred, anchor, aux = model(*args)
            self.assertEqual(tuple(pred.shape), (1, 2, size, size))
            total, _ = compute_losses(pred, anchor, aux, args[1][0][0].long(), {})
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertGreater(float(model.encoder.conv.weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.cacm.adapter[-1].weight.grad.abs().sum()), 0)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_gradient_reaches_query_scale(self):
        model = make_model().train()
        args = episode()
        pred, anchor, aux = model(*args)
        self.assertGreater(float(aux["reverse_loss"]), 0)
        loss, _ = compute_losses(pred, anchor, aux, args[1][0][0].long(), {})
        loss.backward()
        self.assertGreater(float(model.raqa.query_scale_raw.grad.abs()), 0)

    def test_invalid_support_rejected_before_encoder(self):
        model = make_model()
        args = episode()
        args[1][0][0].zero_()
        args[2][0][0].fill_(1)
        with patch.object(model.encoder, "forward", side_effect=AssertionError("must not encode")):
            with self.assertRaises(InvalidSupportEpisode):
                model(*args)

    def test_tiny_foreground_not_skipped(self):
        model = make_model().eval()
        args = episode()
        args[1][0][0].zero_()
        args[1][0][0][0, 1, 1] = 1
        args[2][0][0].copy_(1 - args[1][0][0])
        with torch.no_grad():
            pred, _, _ = model(*args)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(float(model._last_care["alpha"]), 0.)

    def test_multiple_shots_and_ignored_region(self):
        args = episode(shots=3)
        for fg, bg in zip(args[1][0], args[2][0]):
            fg[:, :8] = 0
            bg[:, :8] = 0
        with torch.no_grad():
            pred = make_model().eval()(*args)[0]
        self.assertTrue(torch.isfinite(pred).all())

    def test_query_labels_never_change_prediction(self):
        model, args = make_model().eval(), episode()
        with torch.no_grad():
            a = model(*args, query_gt=torch.zeros(1, 96, 96))[0]
            b = model(*args, query_gt=torch.ones(1, 96, 96))[0]
        self.assertTrue(torch.equal(a, b))

    def test_no_test_time_parameter_or_bn_mutation(self):
        model, args = make_model().eval(), episode()
        before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            model(*args)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))

    def test_class_swap_symmetry(self):
        model, args = make_model().eval(), episode()
        with torch.no_grad():
            a = model(*args)[0]
            b = model(args[0], args[2], args[1], args[3])[0]
        torch.testing.assert_close(a, b.flip(1), atol=2e-5, rtol=2e-5)

    def test_semantic_ablation_is_exact_anchor(self):
        model = make_model(use_competition=False, use_adaptation=False).eval()
        with torch.no_grad():
            pred, anchor, _ = model(*episode())
        self.assertTrue(torch.equal(pred, anchor))

    def test_chunking_equivalence(self):
        a, b = make_model(chunk_size=17).eval(), make_model(chunk_size=512).eval()
        b.load_state_dict(a.state_dict())
        with torch.no_grad():
            x, y = a(*episode())[0], b(*episode())[0]
        torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)

    def test_empty_query_has_useful_gradient(self):
        logits = torch.zeros(1, 2, 8, 8, requires_grad=True)
        loss = segmentation_loss(logits, torch.zeros(1, 8, 8).long())
        loss.backward()
        self.assertGreater(float(logits.grad[:, 1].sum()), 0)

    def test_checkpoint_strict_roundtrip(self):
        model = make_model().eval()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "weights.pth"
            torch.save({"state_dict": model.state_dict()}, path)
            other = CARENet(cfg=model.config, encoder=ToyEncoder(), pretrained_path=path).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(*episode())[0], other(*episode())[0])

    def test_batchnorm_frozen_training(self):
        model = make_model().train()
        self.assertTrue(model.training)
        self.assertFalse(model.encoder.bn.training)


class AuditTest(unittest.TestCase):
    def test_real_forward_accepts_helpful_query_evidence(self):
        torch.manual_seed(31)
        banks = []
        for i in (0, 1):
            z = torch.randn(32, 16) * 2.5
            z[:, 0] += 5 * (2 * i - 1)
            banks.append({"z": F.normalize(z, dim=1), "fold": torch.arange(32) % 2})
        q = torch.zeros(1, 16, 8, 8)
        q[:, 0, :, :4], q[:, 0, :, 4:] = -1, 1
        matcher = CounterClassAnchorCompetition(channels=16, hidden=32, topk=12)
        module = RiskAuditedQueryAdaptation()
        initial = matcher.score(q.permute(0, 2, 3, 1).reshape(-1, 16), matcher.fit(banks))
        final, info = module(q, banks, initial, matcher)
        self.assertGreater(float(info["alpha"]), 0)
        self.assertGreater(float((final - initial).abs().max()), 0)
        risks = info["audit_risks"]
        selected = torch.where(module.candidates == info["alpha"])[0].item()
        self.assertTrue(((risks[:, 0] - risks[:, selected]) >= module.audit_margin).all())

    def test_rejects_improvement_that_hurts_one_fold(self):
        module = RiskAuditedQueryAdaptation()
        risks = torch.tensor([[.5, .4, .3, .2], [.5, .51, .52, .53]])
        alpha, _ = module.select_alpha(risks)
        self.assertEqual(float(alpha), 0)

    def test_accepts_consistent_improvement(self):
        module = RiskAuditedQueryAdaptation()
        risks = torch.tensor([[.5, .4, .3, .2], [.5, .45, .4, .35]])
        alpha, _ = module.select_alpha(risks)
        self.assertEqual(float(alpha), .75)

    def test_no_forced_foreground_query_bank(self):
        module = RiskAuditedQueryAdaptation()
        z = torch.randn(1, 16, 8, 8)
        self.assertIsNone(module.query_bank(z, torch.full((64,), -10.)))

    def test_audit_crossfit_key_provenance(self):
        matcher = CounterClassAnchorCompetition(channels=4, hidden=8, block_size=2)
        torch.manual_seed(4)
        banks = [{"z": F.normalize(torch.randn(12, 4), dim=1),
                  "fold": torch.arange(12) % 2} for _ in range(2)]
        module = RiskAuditedQueryAdaptation(confidence=.51, min_mass=.01)
        z = F.normalize(torch.randn(1, 4, 8, 8), dim=1)
        captured = []
        original = matcher.fit
        def record(fold_banks):
            captured.append(fold_banks)
            return original(fold_banks)
        with patch.object(matcher, "fit", side_effect=record):
            module.audit(z, banks, matcher)
        self.assertEqual(len(captured), 2)
        for fold, fitted in enumerate(captured):
            for i in (0, 1):
                torch.testing.assert_close(fitted[i]["z"], banks[i]["z"][banks[i]["fold"] == 1 - fold])


class InterfaceTest(unittest.TestCase):
    def test_resnet_constructor_dispatch(self):
        backbone = types.ModuleType("models.backbone")
        provider = types.ModuleType("models.backbone.torchvision_backbones")
        calls = []
        def constructor(name):
            def build(coco):
                calls.append((name, coco))
                return ToyEncoder()
            return build
        provider.TVDeeplabRes50Encoder = constructor("resnet50")
        provider.TVDeeplabRes101Encoder = constructor("resnet101")
        with patch.dict(sys.modules, {"models.backbone": backbone,
                                      "models.backbone.torchvision_backbones": provider}):
            for name in ("resnet50", "dlfcn_res50", "resnet101", "dlfcn_res101"):
                cfg = {"which_model": name, "use_coco_init": False,
                       "care": {"feat_dim": 16, "adapter_hidden": 32}}
                CARENet(cfg=cfg)
        self.assertEqual(calls, [("resnet50", False)] * 2 + [("resnet101", False)] * 2)

    def test_paired_comparison_and_protocol_mismatch(self):
        row = {"scan": "1", "class": 1, "organ": "A", "support_scan": "1",
               "support_slice": 2, "query_slices": [1, 3], "slice_mean_dsc": .7}
        old = {"domains": {"SABS": {"protocol": "same_scan_fg", "records": [row]}}}
        new = copy.deepcopy(old)
        new["domains"]["SABS"]["records"][0]["slice_mean_dsc"] = .73
        rows = compare_pair(old, new)
        self.assertAlmostEqual(rows[-1]["delta_points"], 3)
        new["domains"]["SABS"]["records"][0]["query_slices"] = [0, 3]
        with self.assertRaisesRegex(ValueError, "sampling differs"):
            compare_pair(old, new)

    def test_python38_syntax(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 8))

    def test_shell_six_directions_and_backbones(self):
        for backbone in ("resnet50", "resnet101"):
            env = dict(os.environ, DRY_RUN="1", BACKBONE=backbone, EXPERIMENTS="ALL")
            r = subprocess.run(["bash", str(ROOT / "run_care.sh")], env=env,
                               capture_output=True, text=True, check=True)
            self.assertEqual(r.stdout.count("-u train_care.py"), 6)
            self.assertEqual(r.stdout.count("-u test_care.py"), 6)
            self.assertIn("dlfcn_res" + backbone[len("resnet"):], r.stdout)

    def test_shell_single_multi_case_alias(self):
        for tasks, expected in (("ct2mri", 1), ("LGE2bSSFP,bSSFP2LGE", 2),
                                ("CT2MRI,NCI2UCLH,UCLH2NCI", 3)):
            env = dict(os.environ, DRY_RUN="1", EXPERIMENTS=tasks)
            r = subprocess.run(["bash", str(ROOT / "run_care.sh")], env=env,
                               capture_output=True, text=True, check=True)
            self.assertEqual(r.stdout.count("-u train_care.py"), expected)

    def test_organ_aggregation_exact(self):
        records = [{"class": 1, "slice_mean_dsc": .5}, {"class": 1, "slice_mean_dsc": .9},
                   {"class": 2, "slice_mean_dsc": .8}]
        rows = organ_statistics(records, [1, 2, 3], {1: "A", 2: "B", 3: "C"})
        self.assertAlmostEqual(rows["1"]["mean_dsc"], .7)
        self.assertAlmostEqual(rows["1"]["std"], .2)
        self.assertEqual(rows["1"]["N"], 2)
        self.assertIsNone(rows["3"]["mean_dsc"])


if __name__ == "__main__":
    unittest.main()
