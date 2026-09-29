"""Synthetic integration only: Sacred -> original-shaped loader -> checkpoint
-> NIfTI evaluator. The user's real backbone and dataset are NOT available.
"""
import importlib.util
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from care_compat import load_checkpoint


class TinyEncoder(nn.Module):
    def __init__(self, use_coco=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
    def forward(self, x, low_level=False):
        return self.conv(x)


class TinyDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
    def __len__(self):
        return 2
    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        x = torch.randn(3, 24, 24, generator=generator)
        y = torch.zeros(24, 24)
        y[8:17, 7:16] = 1
        support_y = torch.zeros_like(y) if index == 0 else y
        return {"support_images": [[x]], "support_fg_labels": [[support_y]],
                "query_images": [x + .1], "query_labels": [y.long()]}
    def reload_buffer(self):
        pass


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class IntegrationTest(unittest.TestCase):
    def test_train_save_evaluate_all_protocols(self):
        try:
            import sacred
            import SimpleITK as sitk
        except ImportError:
            self.skipTest("Optional integration dependencies: sacred and SimpleITK")
        dataloaders = types.ModuleType("dataloaders")
        dataset_mod = types.ModuleType("dataloaders.datasets")
        dataset_mod.TrainDataset = TinyDataset
        backbone = types.ModuleType("models.backbone")
        encoder_mod = types.ModuleType("models.backbone.torchvision_backbones")
        encoder_mod.TVDeeplabRes101Encoder = TinyEncoder
        encoder_mod.TVDeeplabRes50Encoder = TinyEncoder
        substitutes = {"dataloaders": dataloaders, "dataloaders.datasets": dataset_mod,
                       "models.backbone": backbone,
                       "models.backbone.torchvision_backbones": encoder_mod}
        with patch.dict(sys.modules, substitutes), tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            data.mkdir()
            train = load_module("care_integration_train", ROOT / "train_care.py")
            updates = {"n_steps": 2, "max_iters_per_load": 1, "num_workers": 0,
                       "ce_class_weights": [.2, 1.],
                       "print_interval": 1, "checkpoint_interval": 1,
                       "path": {"log_dir": str(root / "runs"), "SABS": {"data_dir": str(data)}},
                       "model": {"which_model": "dlfcn_res50", "use_coco_init": False,
                                 "care": {"feat_dim": 16, "adapter_hidden": 32,
                                          "audit_block_size": 2}}}
            run = train.ex.run(config_updates=updates)
            checkpoint = Path(run.result)
            saved = load_checkpoint(checkpoint)
            self.assertEqual(saved["step"], 2)
            self.assertEqual(saved["model_config"]["care"]["revision"], 2)
            self.assertAlmostEqual(saved["train_config"]["ce_class_weights"][0], .2)
            self.assertEqual(saved["scheduler"]["last_epoch"], 2)
            self.assertGreater(run.info["support_episode_sampling"]["skipped"], 0)
            self.assertEqual(run.info["support_episode_sampling"]["successful_updates"], 2)
            self.assertTrue((checkpoint.parent / "latest.pth").exists())
            rng = np.random.default_rng(41)
            for sid in (1, 2):
                image = rng.normal(size=(4, 24, 24)).astype(np.float32)
                labels = np.zeros((4, 24, 24), np.uint8)
                labels[1:3, 7:17, 8:18] = 1
                sitk.WriteImage(sitk.GetImageFromArray(image), str(data / ("image_%d.nii.gz" % sid)))
                sitk.WriteImage(sitk.GetImageFromArray(labels), str(data / ("label_%d.nii.gz" % sid)))
            evaluator = load_module("care_integration_eval", ROOT / "test_care.py")
            for protocol in ("same_scan_fg", "same_scan_all", "cross_scan_all"):
                result = evaluator.ex.run(config_updates={
                    "reload_model_path": str(checkpoint), "eval_domains": ["SABS"],
                    "protocol": protocol, "support_scan_id": "1",
                    "result_dir": str(root / "results"),
                    "path": {"SABS": {"data_dir": str(data), "test_label": [1], "target_size": 24}}})
                summary = json.loads(Path(result.result).read_text())["domains"]["SABS"]
                self.assertEqual(summary["protocol"], protocol)
                if protocol == "cross_scan_all":
                    self.assertEqual(summary["N"], 1)
                    self.assertEqual(summary["records"][0]["scan"], "2")
                    self.assertEqual(summary["records"][0]["support_scan"], "1")
                    self.assertEqual(summary["empty_query_slices"], 2)
                elif protocol == "same_scan_all":
                    self.assertEqual(summary["empty_query_slices"], 4)
                else:
                    self.assertEqual(summary["empty_query_slices"], 0)
                self.assertTrue(0 <= summary["legacy_scan_class_mean_dsc"] <= 1)
                self.assertIn("organ_statistics", summary)
                self.assertTrue(Path(result.result).with_suffix(".csv").is_file())
            with self.assertRaisesRegex(ValueError, "training source"):
                evaluator.ex.run(config_updates={"reload_model_path": str(checkpoint),
                                                  "expected_source": "CHAOST2"})


if __name__ == "__main__":
    unittest.main()
