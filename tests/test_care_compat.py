"""Metadata and state-key migration checks independent of the GPU stack."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from care_compat import model_config_from_checkpoint, model_state_from_checkpoint


class CheckpointNamingTest(unittest.TestCase):
    def test_historical_metadata_and_module_names(self):
        old = "historical_model"
        checkpoint = {
            "model_config": {"which_model": "dlfcn_res50", old: {"feat_dim": 16, "adapter_hidden": 32}},
            "state_dict": {"module.encoder.weight": 1, "module.ccm.scale_raw": 2,
                           "module.sara.query_scale_raw": 3},
        }
        config = model_config_from_checkpoint(checkpoint)
        self.assertNotIn(old, config)
        self.assertEqual(config["care"]["feat_dim"], 16)
        self.assertEqual(model_state_from_checkpoint(checkpoint), {
            "encoder.weight": 1, "cacm.scale_raw": 2, "raqa.query_scale_raw": 3})

    def test_current_names_and_duplicate_detection(self):
        state = {"state_dict": {"cacm.a": 1, "raqa.b": 2}}
        self.assertEqual(model_state_from_checkpoint(state), state["state_dict"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            model_state_from_checkpoint({"state_dict": {"ccm.a": 1, "cacm.a": 2}})


if __name__ == "__main__":
    unittest.main()
