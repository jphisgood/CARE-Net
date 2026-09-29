import glob
import os
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from copy import deepcopy
from care_defaults import MODEL, PATHS

sacred.SETTINGS["CONFIG"]["READ_ONLY_CONFIG"] = False
sacred.SETTINGS.CAPTURE_MODE = "no"
ex = Experiment("CARE_NET")
ex.captured_out_filter = apply_backspaces_and_linefeeds
for pattern in ("*.py", "models/*.py", "models/backbone/*.py", "dataloaders/*.py", "util/*.py"):
    for source in glob.glob(pattern):
        ex.add_source_file(source)


@ex.config
def cfg():
    seed = 1234
    gpu_id = 0
    num_workers = 8
    dataset = "SABS"
    test_label = [1, 2, 3, 6]
    exclude_label = None
    n_sv = 5000
    min_size = 200
    use_gt = False
    batch_size = 1
    ignore_label = 255
    # Reuse the original project's WCE convention; do not guess its weights.
    use_wce = True
    ce_class_weights = None  # explicit [background, foreground] overrides project lookup
    n_steps = 40000
    max_iters_per_load = 1000
    print_interval = 100
    checkpoint_interval = 1000
    gradient_clip_norm = 5.0
    lambda_dice = 0.5
    lambda_anchor = 0.5
    lambda_competitive = 0.25
    lambda_reverse = 0.1
    model = deepcopy(MODEL)
    model["care"]["revision"] = 2
    task = {"n_ways": 1, "n_shots": 1, "n_queries": 1}
    optim = {"lr": 1e-3, "backbone_lr_mult": 0.10,
             "momentum": 0.9, "weight_decay": 5e-4}
    lr_step_every = 1000
    lr_step_gamma = 0.95
    run_prefix = ""
    path = deepcopy(PATHS)
    path["log_dir"] = "./runs"


@ex.config_hook
def add_observer(config, command_name, logger):
    del command_name, logger
    name = "CARE_NET_%s_%s_%dshot" % (config["run_prefix"], config["dataset"], config["task"]["n_shots"])
    ex.observers.append(FileStorageObserver.create(os.path.join(config["path"]["log_dir"], name)))
    return config
