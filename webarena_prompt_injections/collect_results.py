# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Join the per-cell artifacts of one WASP sub-run (<output-dir>/<idx>/) into a single
per-template results table: results_per_template.json and results_per_template.csv.

One row per template (= per task_id). Each row is keyed by a globally-unique, human
readable `record_id` so rows never collapse when concatenated across grid cells that all
reuse task_ids starting at 1000.

Sources joined (all inside the sub-run dir):
  template_index.json                 task_id -> template metadata (free_form_name, exfil, ...)
  cell_meta.json                      config_stem / model / user_goal_idx / injection_format
  step_by_step_asr.json               per_task step-by-step verdict (compromised / not / refusal)
  attacker_utility.json               per_task attacker end-to-end success (0/1)
  user_utility.json                   per_task benign user-task success (0/1)
  agent_logs/{task_id}.meta.json      per-task runner provenance (crashed, timing, steps)

Usable as a module (from run.py) via collect(run_dir) or standalone:
  python collect_results.py --run-dir <output-dir>/<idx>
"""

import csv
import json
import os

import click

from utils import read_json, slug, write_json


# Column order for the CSV (record_id first so a row is self-identifying).
_COLUMNS = [
    "record_id",
    "config_stem",
    "model",
    "output_format",
    "user_goal_idx",
    "injection_format",
    "task_id",
    "environment",
    "free_form_name",
    "exfil",
    "step_by_step",     # compromised / not_compromised / outright_refusal
    "attacker_success", # 0/1 end-to-end attacker goal achieved
    "user_success",     # 0/1 benign user goal achieved
    "crashed",
    "num_steps",
    "duration_s",
    "attempts",
    "user_intent",
    "attacker_intent",
    "trace_path",
    "meta_path",
    "user_task_path",
    "attacker_task_path",
]


def _maybe_read_json(path):
    """Read a JSON file, or return None if it is missing/unreadable (a partial run may
    lack some artifacts; the row should still be emitted with blanks)."""
    if not os.path.isfile(path):
        return None
    try:
        return read_json(path)
    except Exception as e:
        print(f"[collect_results] WARNING: could not read {path}: {e}", flush=True)
        return None


def build_record_id(cell, environment, task_id, free_form_name):
    return "__".join([
        cell.get("config_stem", "cfg"),
        environment or "env",
        f"ug{cell.get('user_goal_idx', '?')}",
        cell.get("injection_format", "injfmt"),
        f"t{task_id}",
        slug(free_form_name),
    ])


def collect(run_dir, cell_overrides=None):
    """Build the per-template rows for one sub-run and write results_per_template.{json,csv}.

    cell_overrides lets run.py supply config_stem/user_goal_idx/injection_format when a
    fresh run has them in memory; otherwise they are read from cell_meta.json.
    Returns the list of row dicts (also written to disk).
    """
    template_index = _maybe_read_json(os.path.join(run_dir, "template_index.json")) or {}
    cell = _maybe_read_json(os.path.join(run_dir, "cell_meta.json")) or {}
    if cell_overrides:
        cell = {**cell, **{k: v for k, v in cell_overrides.items() if v is not None}}

    sbs = _maybe_read_json(os.path.join(run_dir, "step_by_step_asr.json")) or {}
    sbs_per_task = sbs.get("per_task", {}) or {}

    attacker = _maybe_read_json(os.path.join(run_dir, "attacker_utility.json")) or {}
    user = _maybe_read_json(os.path.join(run_dir, "user_utility.json")) or {}
    attacker_per_task = {str(e["task_id"]): e.get("score") for e in attacker.get("per_task", []) or []}
    user_per_task = {str(e["task_id"]): e.get("score") for e in user.get("per_task", []) or []}

    log_dir = os.path.join(run_dir, "agent_logs")

    rows = []
    for task_id in sorted(template_index, key=lambda k: int(k)):
        tmpl = template_index[task_id]
        environment = tmpl.get("environment", "")
        meta = _maybe_read_json(os.path.join(log_dir, f"{task_id}.meta.json")) or {}

        trace_path = os.path.join(log_dir, f"{task_id}.jsonl")
        row = {
            "record_id": build_record_id(cell, environment, task_id, tmpl.get("free_form_name", "")),
            "config_stem": cell.get("config_stem", ""),
            # Prefer the per-task runtime-captured backbone (meta.json) over the cell's
            # requested model: for PTE-backed agents the WASP --model flag is ignored by
            # the runner, so cell["model"] can mislabel the run (e.g. gpt-4o vs the real
            # claude-sonnet-4-6). meta["model"] is written by the runner from PTE config.
            "model": meta.get("model") or cell.get("model", ""),
            "output_format": cell.get("output_format", ""),
            "user_goal_idx": cell.get("user_goal_idx", ""),
            "injection_format": cell.get("injection_format", ""),
            "task_id": int(task_id),
            "environment": environment,
            "free_form_name": tmpl.get("free_form_name", ""),
            "exfil": tmpl.get("exfil", ""),
            "step_by_step": sbs_per_task.get(str(task_id), ""),
            "attacker_success": attacker_per_task.get(str(task_id), ""),
            "user_success": user_per_task.get(str(task_id), ""),
            "crashed": meta.get("crashed", ""),
            "num_steps": meta.get("num_steps", ""),
            "duration_s": meta.get("duration_s", ""),
            "attempts": meta.get("attempts", ""),
            "user_intent": tmpl.get("user_intent", ""),
            "attacker_intent": tmpl.get("attacker_intent", ""),
            "trace_path": trace_path if os.path.isfile(trace_path) else "",
            "meta_path": os.path.join(log_dir, f"{task_id}.meta.json") if meta else "",
            "user_task_path": os.path.join(run_dir, "webarena_tasks", f"{task_id}.json"),
            "attacker_task_path": os.path.join(run_dir, "webarena_tasks_attacker", f"{task_id}.json"),
        }
        rows.append(row)

    write_json(rows, os.path.join(run_dir, "results_per_template.json"))
    write_csv(rows, os.path.join(run_dir, "results_per_template.csv"))
    print(f"[collect_results] Wrote {len(rows)} template rows to {run_dir}", flush=True)
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@click.command()
@click.option("--run-dir", required=True, help="Path to one sub-run directory (<output-dir>/<idx>)")
def main(run_dir):
    collect(run_dir)


if __name__ == "__main__":
    main()
