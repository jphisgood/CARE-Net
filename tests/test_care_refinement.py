"""R2 regression: partial-volume support, boundary audit, WCE and compatibility."""
import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_care_model import make_model, episode
from models.anchor_competition import CounterClassAnchorCompetition
from models.query_adaptation import RiskAuditedQueryAdaptation
from losses_care import resolve_class_weights, segmentation_loss, compute_losses
from test_care import checkpoint_model_config


class RefinementTest(unittest.TestCase):
    def test_thin_support_keeps_multiple_local_descriptors(self):
        torch.manual_seed(81)
        z = F.normalize(torch.randn(1, 16, 8, 8), dim=1)
        fg = torch.zeros(1, 1, 8, 8)
        fg[:, :, :, 3] = .15
        old = CounterClassAnchorCompetition(16, 32, revision=1)
        new = CounterClassAnchorCompetition(16, 32, revision=2)
        a, b = old.support_banks(z, fg, 1 - fg), new.support_banks(z, fg, 1 - fg)
        self.assertEqual(len(a[1]["z"]), 1)
        self.assertGreater(len(b[1]["z"]), 1)
        self.assertTrue(torch.isfinite(b[1]["z"]).all())

    def test_all_native_partial_cells_have_regional_representation(self):
        z = F.normalize(torch.randn(1, 16, 8, 8), dim=1)
        fg = torch.full((1, 1, 8, 8), .2)
        matcher = CounterClassAnchorCompetition(16, 32, revision=2)
        bank = matcher.support_banks(z, fg, 1 - fg)
        self.assertEqual(len(bank[1]["z"]), 16)
        self.assertTrue((bank[1]["quality"] < .5).all())

    def test_weighted_pooling_ignores_heldout_features(self):
        torch.manual_seed(37)
        z = F.normalize(torch.randn(1, 8, 8, 8), dim=1)
        fg = torch.full((1, 1, 8, 8), .2)
        keep = torch.zeros_like(fg)
        keep[:, :, ::2, :] = 1
        matcher = CounterClassAnchorCompetition(8, 16, revision=2)
        before = matcher.support_banks(z, fg * keep, (1 - fg) * keep)
        changed = z.clone()
        changed[:, :, 1::2, :] = 1000 * torch.randn_like(changed[:, :, 1::2, :])
        after = matcher.support_banks(changed, fg * keep, (1 - fg) * keep)
        for i in (0, 1):
            torch.testing.assert_close(before[i]["z"], after[i]["z"])

    def test_boundary_risk_vetoes_better_region_score(self):
        module = RiskAuditedQueryAdaptation(revision=2)
        risks = torch.tensor([[.5, .4, .3, .2], [.6, .5, .4, .3]])
        boundary = torch.tensor([[.4, .41, .42, .43], [.4, .41, .42, .43]])
        alpha, _ = module.select_region_alpha(risks, boundary)
        self.assertEqual(float(alpha), 0)

    def test_region_and_boundary_improvement_can_be_accepted(self):
        module = RiskAuditedQueryAdaptation(revision=2)
        risks = torch.tensor([[.5, .4, .3, .2], [.6, .5, .4, .3]])
        boundary = torch.tensor([[.4, .39, .38, .37], [.4, .39, .38, .37]])
        alpha, _ = module.select_region_alpha(risks, boundary)
        self.assertEqual(float(alpha), .75)

    def test_audit_uses_heldout_soft_labels(self):
        torch.manual_seed(8)
        z = F.normalize(torch.randn(1, 8, 8, 8), dim=1)
        fg = torch.full((1, 1, 8, 8), .2)
        matcher = CounterClassAnchorCompetition(8, 16, revision=2, block_size=2)
        banks = matcher.support_banks(z, fg, 1 - fg)
        module = RiskAuditedQueryAdaptation(revision=2)
        received = []
        original = module.region_risk
        def record(logit, y, weight):
            received.append(y.clone())
            return original(logit, y, weight)
        fake_bank = [F.normalize(torch.randn(4, 8), dim=1) for _ in range(2)]
        with patch.object(module, "query_bank", return_value=fake_bank), \
             patch.object(module, "region_risk", side_effect=record):
            module.audit(z, banks, matcher)
        self.assertEqual(len(received), 8)
        for y in received:
            torch.testing.assert_close(y, torch.full_like(y, .2))

    def test_old_state_dict_loads_strictly_in_both_revisions(self):
        old, new = make_model(revision=1), make_model(revision=2)
        new.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        with torch.no_grad():
            self.assertTrue(torch.isfinite(new.eval()(*episode())[0]).all())

    def test_metadata_preserves_old_behavior_unless_explicit(self):
        old_config = copy.deepcopy(make_model().config)
        old_config["care"].pop("revision", None)
        checkpoint = {"model_config": old_config}
        self.assertEqual(checkpoint_model_config(checkpoint, {})["care"]["revision"], 1)
        self.assertEqual(checkpoint_model_config(checkpoint, {"care": {"revision": 2}})["care"]["revision"], 2)

    def test_project_weight_resolver_calls_original_function(self):
        package, module = types.ModuleType("util"), types.ModuleType("util.utils")
        calls = []
        def resolve(flag, dataset):
            calls.append((flag, dataset))
            return torch.tensor([.25, 1.])
        module.compose_wt_simple = resolve
        with patch.dict(sys.modules, {"util": package, "util.utils": module}):
            weight = resolve_class_weights({"use_wce": True, "dataset": "SABS"}, "cpu")
        self.assertEqual(calls, [(True, "SABS")])
        torch.testing.assert_close(weight, torch.tensor([.25, 1.]))

    def test_weighted_loss_matches_torch(self):
        logits = torch.randn(1, 2, 5, 5)
        target = torch.zeros(1, 5, 5).long()
        target[:, 2:4, 2:4] = 1
        target[:, 0] = 255
        actual = segmentation_loss(logits, target, dice_weight=0., class_weights=[.2, 1.])
        expected = F.cross_entropy(logits, target, ignore_index=255, weight=torch.tensor([.2, 1.]))
        torch.testing.assert_close(actual, expected)

    def test_refinement_shapes_backward_and_query_label_independence(self):
        for size in (192, 256, 257):
            model = make_model(revision=2, feat_dim=256, adapter_hidden=128).train()
            args = episode(size)
            prediction, anchor, aux = model(*args)
            loss, _ = compute_losses(prediction, anchor, aux, args[1][0][0].long(), {"ce_class_weights": [.2, 1.]})
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(tuple(prediction.shape), (1, 2, size, size))
            self.assertGreater(float(model.encoder.conv.weight.grad.abs().sum()), 0)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            model.eval()
            with torch.no_grad():
                a = model(*args, query_gt=torch.zeros_like(args[1][0][0]))[0]
                b = model(*args, query_gt=torch.ones_like(args[1][0][0]))[0]
            self.assertTrue(torch.equal(a, b))

    def test_refinement_multiple_shots_and_single_pixel(self):
        model = make_model(revision=2).eval()
        for tiny in (False, True):
            args = episode(shots=2)
            if tiny:
                for fg, bg in zip(args[1][0], args[2][0]):
                    fg.zero_()
                    fg[:, 1, 1] = 1
                    bg.copy_(1 - fg)
            with torch.no_grad():
                self.assertTrue(torch.isfinite(model(*args)[0]).all())

    def test_refinement_class_swap_and_eval_state(self):
        model = make_model(revision=2).eval()
        args = episode()
        before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            a = model(*args)[0]
            b = model(args[0], args[2], args[1], args[3])[0]
        torch.testing.assert_close(a, b.flip(1), atol=2e-5, rtol=2e-5)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))

    def test_query_disagreement_suppresses_actual_update(self):
        model = make_model(revision=2).eval()
        args = episode()
        pixels = 12 * 12  # ToyEncoder produces ceil(96/8) feature grid.
        audit_info = {"audit_before": torch.tensor(.4), "audit_after": torch.tensor(.3),
                      "query_agreement": torch.zeros(pixels)}
        with patch.object(model.raqa, "audit", return_value=(torch.tensor(.75), audit_info, "accepted")) as audit:
            with torch.no_grad():
                pred, _, aux = model(*args)
        self.assertTrue(audit.called)
        torch.testing.assert_close(pred, aux["competitive_logits"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
