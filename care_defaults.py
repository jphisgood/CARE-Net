"""One source of model defaults for train/evaluation."""
from copy import deepcopy

MODEL = {
    "use_coco_init": True, "which_model": "dlfcn_res101",
    "care": {"revision": 2, "feat_dim": 256, "adapter_hidden": 128, "topk": 12,
             "opponent_topk": 8, "max_support_tokens": 1024, "chunk_size": 256,
             "max_feature_size": 64, "freeze_backbone_bn": True,
             "initial_competition": .15, "use_competition": True,
             "query_grid": 8, "query_topk": 3, "query_confidence": .65,
             "query_min_mass": 1., "correction_cap": 4.,
             "audit_block_size": 4, "audit_margin": .002,
             "use_adaptation": True, "use_audit": True},
}
PATHS = {
    "SABS": {"data_dir": "./data/ABD/ABDOMEN_CT/sabs_CT_normalized", "test_label": [6, 2, 3, 1], "target_size": 256},
    "CHAOST2": {"data_dir": "./data/ABD/ABDOMEN_MR/chaos_MR_T2_normalized", "test_label": [1, 2, 3, 4], "target_size": 257},
    "CARDIAC_bssFP": {"data_dir": "./data/Cardiac/bSSFP/cmr_bssFP_normalized", "test_label": [1, 2, 3], "target_size": 192},
    "CARDIAC_LGE": {"data_dir": "./data/Cardiac/LGE/cmr_LGE_normalized", "test_label": [1, 2, 3], "target_size": 192},
    "Prostate_NCI": {"data_dir": "./data/Prostate/NCI/NCI_normalized", "test_label": [1, 5, 6], "target_size": 192},
    "Prostate_UCLH": {"data_dir": "./data/Prostate/UCLH/UCLH_normalized", "test_label": [1, 5, 6], "target_size": 192},
}


def merge_config(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
