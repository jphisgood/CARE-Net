"""Pair evaluation records only when they use the same support and query slices."""


def compare_pair(reference, candidate):
    domains_a, domains_b = reference["domains"], candidate["domains"]
    if set(domains_a) != set(domains_b):
        raise ValueError("Evaluated domains differ")
    rows = []
    for domain in sorted(domains_a):
        a, b = domains_a[domain], domains_b[domain]
        if a["protocol"] != b["protocol"]:
            raise ValueError("Evaluation sampling differs: protocol")
        def indexed(result):
            records = {(str(r["scan"]), int(r["class"])): r for r in result["records"]}
            if len(records) != len(result["records"]):
                raise ValueError("Duplicate scan/class records")
            return records
        left, right = indexed(a), indexed(b)
        if set(left) != set(right):
            raise ValueError("Evaluation sampling differs: scan/class records")
        for key in sorted(left):
            x, y = left[key], right[key]
            for field in ("support_scan", "support_slices", "query_slices"):
                if x.get(field) != y.get(field):
                    raise ValueError("Evaluation sampling differs: " + field)
            delta = 100 * (y["slice_mean_dsc"] - x["slice_mean_dsc"])
            rows.append({"domain": domain, "scan": key[0], "class": key[1],
                         "reference_dsc": x["slice_mean_dsc"],
                         "candidate_dsc": y["slice_mean_dsc"],
                         "delta_points": delta})
    if not rows:
        raise ValueError("No paired scan/class records")
    rows.append({"domain": "OVERALL", "scan": "", "class": "",
                 "reference_dsc": sum(r["reference_dsc"] for r in rows) / len(rows),
                 "candidate_dsc": sum(r["candidate_dsc"] for r in rows) / len(rows),
                 "delta_points": sum(r["delta_points"] for r in rows) / len(rows)})
    return rows
