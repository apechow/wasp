"""One-off: rebuild top-level results_all.csv + summary.json for a completed run
directory by re-reading every cell subdir. Mirrors run.run_single_end_to_end's
metric accumulation (lines ~112-153) without re-running any agent, so a cell that
was recovered out-of-band (e.g. cell 0 after a Playwright timeout) folds back into
the aggregate. Format is identical because we reuse run._write_run_summary and
collect_results.collect."""
import json
import os
import sys
from collections import defaultdict

import collect_results
import run as run_mod


def _read_json_or_none(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def restitch(output_dir):
    manifest = json.load(open(os.path.join(output_dir, "run_manifest.json")))
    config = manifest["config"]
    output_format = manifest["output_format"]
    model = manifest["requested_model"]  # the --model flag as originally passed
    cells = manifest["cells"]

    results_dict = defaultdict(int)
    all_rows = []

    for idx in sorted(cells, key=int):
        cell = cells[idx]
        subdir = os.path.join(output_dir, idx)
        if not os.path.isdir(subdir):
            print(f"WARNING: missing cell subdir {subdir}, skipping")
            continue

        sbs = _read_json_or_none(os.path.join(subdir, "step_by_step_asr.json"))
        if sbs is not None:
            results_dict["cnt_intermediate_compromised"] += sbs.get("compromised", 0)
            results_dict["cnt_intermediate_not_compromised"] += sbs.get("not_compromised", 0)
            results_dict["cnt_intermediate_outright_refusal"] += sbs.get("outright_refusal", 0)
        else:
            print(f"WARNING: {subdir} step_by_step_asr.json not found")

        att = _read_json_or_none(os.path.join(subdir, "attacker_utility.json"))
        if att is not None:
            results_dict["cnt_end2end_compromised"] += att["total_scores"]
            results_dict["cnt_end2end_not_compromised"] += att["cnt_tasks"] - att["total_scores"]
        else:
            print(f"WARNING: {subdir} attacker_utility.json not found")

        usr = _read_json_or_none(os.path.join(subdir, "user_utility.json"))
        if usr is not None:
            results_dict["cnt_user_utility"] += usr["total_scores"]
            results_dict["cnt_user_total_tasks"] += usr["cnt_tasks"]
        else:
            print(f"WARNING: {subdir} user_utility.json not found")

        try:
            rows = collect_results.collect(subdir, cell_overrides={
                "config_stem": os.path.splitext(os.path.basename(config))[0],
                "model": run_mod.resolve_effective_model(output_format, model),
                "output_format": output_format,
                "user_goal_idx": cell["user_goal_idx"],
                "injection_format": cell["injection_format"],
            })
        except Exception as e:
            print(f"WARNING: collect_results failed for {subdir}: {e}")
            rows = []
        all_rows.extend(rows)
        print(f"cell {idx}: +{len(rows)} rows  (ug={cell['user_goal_idx']}, {cell['injection_format']})")

    run_mod._write_run_summary(output_dir, results_dict, all_rows)
    print(f"\nWrote {len(all_rows)} rows to results_all.csv")
    print("summary.json:")
    print(json.dumps(dict(results_dict), indent=4))


if __name__ == "__main__":
    restitch(sys.argv[1])
