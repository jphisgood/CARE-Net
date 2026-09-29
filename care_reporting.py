"""Per-organ reporting with exactly the original scan/class aggregation.

Prefer the project's get_label_names over any common benchmark mapping.
No scores are inferred from an OVERALL line; old JSON records are required.
"""
import csv
import math
import numpy as np

DOMAIN_NAMES = {"SABS": "ABDOMEN_CT", "CHAOST2": "ABDOMEN_MR"}
# Common preprocessed CHAOS/BTCV labels. Project mappings take precedence.
ABDOMINAL_NAMES = {
    "ABDOMEN_CT": {1: "Spleen", 2: "Right kidney", 3: "Left kidney", 6: "Liver"},
    "ABDOMEN_MR": {1: "Liver", 2: "Right kidney", 3: "Left kidney", 4: "Spleen"},
}


def label_names(domain, config=None):
    config = config or {}
    if config.get("label_names"):
        return {int(k): str(v) for k, v in config["label_names"].items()}, "explicit config"
    mapped = DOMAIN_NAMES.get(domain, domain)
    try:
        from dataloaders.dataset_specifics import get_label_names
    except ModuleNotFoundError as error:
        if error.name not in ("dataloaders", "dataloaders.dataset_specifics"):
            raise
    else:
        names = get_label_names(mapped)
        return {int(k): str(v) for k, v in names.items()}, "project dataset_specifics"
    if mapped in ABDOMINAL_NAMES:
        return dict(ABDOMINAL_NAMES[mapped]), "common benchmark mapping; verify label convention"
    raise ValueError("No verified organ names for %s. Provide path.%s.label_names "
                     "or retain dataloaders/dataset_specifics.py." % (domain, domain))


def organ_statistics(records, requested_labels, names):
    stats = {}
    for label in requested_labels:
        label = int(label)
        if label not in names:
            raise ValueError("Organ name missing for label %d; check dataset mapping" % label)
        values = [float(r["slice_mean_dsc"]) for r in records if int(r["class"]) == label]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
            raise ValueError("Non-finite/out-of-range DSC for label %d" % label)
        stats[str(label)] = {
            "label": label, "organ": names[label],
            "mean_dsc": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "N": len(values),
        }
    return stats


def print_organ_summary(result):
    print("\n---------- Per-organ DSC: %s (%s) ----------" %
          (result["domain"], result["protocol"]))
    print("N = number of evaluated scan/class records; std uses ddof=0.")
    for row in result["organ_statistics"].values():
        if row["N"]:
            print("[ORGAN] Label=%d Organ=%s Mean_DSC=%.6f DSC_percent=%.2f Std=%.6f N=%d" %
                  (row["label"], row["organ"], row["mean_dsc"],
                   100 * row["mean_dsc"], row["std"], row["N"]))
        else:
            print("[ORGAN] Label=%d Organ=%s Mean_DSC=NA Std=NA N=0" % (row["label"], row["organ"]))
    print("[CLASS_MACRO] Mean_DSC=%.6f" % result["class_macro_mean_dsc"])
    # Preserve the legacy score, weighting, precision, and log parser format.
    print("[OVERALL] Mean_DSC=%.6f Std=%.6f N=%d" %
          (result["legacy_scan_class_mean_dsc"], result["std"], result["N"]))


def write_organ_csv(path, domain_results):
    fields = ["domain", "protocol", "label", "organ", "mean_dsc", "dsc_percent", "std", "N", "source_result", "checkpoint"]
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in domain_results.values():
            for row in result["organ_statistics"].values():
                writer.writerow(dict(row, domain=result["domain"], protocol=result["protocol"],
                                     dsc_percent=None if row["mean_dsc"] is None else 100 * row["mean_dsc"],
                                     source_result=result.get("source_result", ""), checkpoint=result.get("checkpoint", "")))
