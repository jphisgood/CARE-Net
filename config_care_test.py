from copy import deepcopy
import sacred
from sacred import Experiment
from care_defaults import PATHS

sacred.SETTINGS["CONFIG"]["READ_ONLY_CONFIG"] = False
sacred.SETTINGS.CAPTURE_MODE = "no"
ex = Experiment("CARE_NET_TEST")


@ex.config
def cfg():
    gpu_id = 0
    seed = 1234
    n_shots = 1
    expected_source = ""  # shell sets this; reject a mismatched source checkpoint
    eval_domains = ["CHAOST2"]
    reload_model_path = "./runs/CARE_NET__SABS_1shot/1/snapshots/40000.pth"
    # Architecture and switches are read from the checkpoint. Explicit entries
    # here override it (useful for inference ablations, not default evaluation).
    model = {}
    use_horizontal_flip_tta = False
    protocol = "same_scan_fg"  # reproduces the supplied evaluator's protocol
    support_scan_id = ""  # REQUIRED for cross_scan_all
    result_dir = "./results/care"
    path = deepcopy(PATHS)
