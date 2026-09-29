"""Compatibility helpers for the user's Python 3.8 / older PyTorch runtime."""
import inspect
import sys


def model_config_from_checkpoint(checkpoint):
    """Normalize model metadata from either naming scheme without changing it in place."""
    raw = checkpoint["model_config"]
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint model_config must be a mapping")
    if "care" in raw:
        settings = raw["care"]
    else:
        candidates = [value for value in raw.values()
                      if isinstance(value, dict) and "feat_dim" in value
                      and "adapter_hidden" in value]
        if len(candidates) != 1:
            raise ValueError("Cannot identify model settings in checkpoint")
        settings = candidates[0]
    if not isinstance(settings, dict):
        raise ValueError("Checkpoint model settings must be a mapping")
    return {**{key: value for key, value in raw.items()
               if key == "care" or not isinstance(value, dict) or value is not settings},
            "care": dict(settings)}


def model_state_from_checkpoint(checkpoint):
    """Rename historical module prefixes while preserving all tensors."""
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state_dict must be a mapping")
    prefixes = {"ccm.": "cacm.", "sara.": "raqa."}
    normalized = {}
    for key, value in state.items():
        name = key[7:] if key.startswith("module.") else key
        for old, new in prefixes.items():
            if name.startswith(old):
                name = new + name[len(old):]
                break
        if name in normalized:
            raise ValueError("Duplicate model parameter after normalization: " + name)
        normalized[name] = value
    return normalized


def load_checkpoint(path, map_location="cpu"):
    """Load a trusted local model checkpoint on old and new torch versions.

    Older torch versions forward unknown kwargs to pickle, so do not pass
    weights_only unless torch.load explicitly supports that argument.
    """
    import torch
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def run_cli(experiment):
    """Keep Sacred logging/configuration; let Python print original errors.

    Sacred --debug only reraises errors here. It does not enter pdb. This
    avoids the broken filtered traceback formatter in some Sacred versions.
    """
    argv = list(sys.argv)
    if "--debug" not in argv and "-d" not in argv:
        argv.insert(1, "--debug")
    return experiment.run_commandline(argv)
