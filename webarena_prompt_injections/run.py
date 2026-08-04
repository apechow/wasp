# Copyright (c) Meta Platforms, Inc. and affiliates.
import click
import csv
import os
import shutil
import subprocess
import sys
import json
from collections import defaultdict
from datetime import datetime, timezone

import collect_results
from constants import (
    PromptInjectionFormat,
    GitlabUserGoals,
    RedditUserGoals
)
from utils import resolve_effective_model


def _subdir_for(output_dir, output_dir_idx):
    sep = "" if output_dir.endswith("/") else "/"
    return f"{output_dir}{sep}{output_dir_idx}/"


def _run_and_tee(command, log_path):
    """Run command, streaming stdout+stderr live to the console AND to log_path, so the
    full per-cell output (agent prints, grader results, tracebacks) is retained rather
    than vanishing with the subprocess. Returns the exit code (does not raise)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        proc.wait()
    return proc.returncode


def _read_json_or_none(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"WARNING: could not read {path}: {e}")
        return None


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
        ).strip()
    except Exception:
        return None


def run_single_end_to_end(
        config,  # prompt injection configurations
        model,  # agentic model to run
        system_prompt,  # system prompt for agent
        output_dir,  # base output dir (subdir is derived from output_dir_idx)
        output_format,  # agentic scaffolding format
        user_goal_idx,  # benign user goal
        injection_format,  # prompt injection format to try
        results_dict,  # to save and accumulate results
        output_dir_idx=0,  # to save logs for each run separately
        multi_docker=False,  # use multi-docker worker pool
    ):
    subdir = _subdir_for(output_dir, output_dir_idx)

    command = [
        'bash',
        'scripts/run_end_to_end.sh',
        subdir,
        model,
        system_prompt,
        config,
        str(user_goal_idx),
        injection_format,
        output_format,
        '--multi-docker' if multi_docker else '',
    ]
    print(f"\nRunning command: \n{' '.join([str(arg) for arg in command])}", flush=True)

    # Tee the whole cell's output to a persistent top-level log (run_end_to_end.sh
    # rm -rf's the subdir at its start, so we cannot write inside it beforehand).
    cell_log = os.path.join(output_dir, "logs", f"cell_{output_dir_idx}.log")
    rc = _run_and_tee(command, cell_log)
    if rc != 0:
        # Do NOT abort the sweep — one cell failing must not discard the others.
        print(f"WARNING: cell {output_dir_idx} exited with code {rc} (continuing)", flush=True)

    # Mirror the cell log into the (now-populated) subdir for self-contained retraceability.
    try:
        shutil.copy(cell_log, os.path.join(subdir, "run.log"))
    except Exception as e:
        print(f"WARNING: could not copy run.log into {subdir}: {e}", flush=True)

    # ------- read metrics from the sub-run dir (was /tmp, which got clobbered across cells)
    res_step_by_step = _read_json_or_none(os.path.join(subdir, "step_by_step_asr.json"))
    if res_step_by_step is not None:
        results_dict["cnt_intermediate_compromised"] += res_step_by_step.get("compromised", 0)
        results_dict["cnt_intermediate_not_compromised"] += res_step_by_step.get("not_compromised", 0)
        results_dict["cnt_intermediate_outright_refusal"] += res_step_by_step.get("outright_refusal", 0)
    else:
        print(f"WARNING: {subdir}step_by_step_asr.json not found — step-by-step ASR metrics skipped")

    res_attacker_utility = _read_json_or_none(os.path.join(subdir, "attacker_utility.json"))
    if res_attacker_utility is not None:
        results_dict["cnt_end2end_compromised"] += res_attacker_utility["total_scores"]
        results_dict["cnt_end2end_not_compromised"] += (
            res_attacker_utility["cnt_tasks"] - res_attacker_utility["total_scores"]
        )
    else:
        print(f"WARNING: {subdir}attacker_utility.json not found — attacker utility metrics skipped")

    res_user_utility = _read_json_or_none(os.path.join(subdir, "user_utility.json"))
    if res_user_utility is not None:
        results_dict["cnt_user_utility"] += res_user_utility["total_scores"]
        results_dict["cnt_user_total_tasks"] += res_user_utility["cnt_tasks"]
    else:
        print(f"WARNING: {subdir}user_utility.json not found — user utility metrics skipped")
    # -------

    # Build the per-template results table for this cell and return its rows so the
    # top-level results_all.csv can concatenate them (keyed by the unique record_id).
    try:
        rows = collect_results.collect(subdir, cell_overrides={
            "config_stem": os.path.splitext(os.path.basename(config))[0],
            # Effective backbone (PTE agents ignore --model); used only as a fallback —
            # collect_results prefers the per-task meta.json model when present.
            "model": resolve_effective_model(output_format, model),
            "output_format": output_format,
            "user_goal_idx": user_goal_idx,
            "injection_format": injection_format,
        })
    except Exception as e:
        print(f"WARNING: collect_results failed for {subdir}: {e}", flush=True)
        rows = []
    return rows


def run_all(config,
            model,
            system_prompt,
            output_dir,
            output_format,
            run_single,
            user_goal_start,
            multi_docker=False):
    gitlab_user_goals = GitlabUserGoals("")
    reddit_user_goals = RedditUserGoals("")
    assert len(gitlab_user_goals.GOALS) == len(reddit_user_goals.GOALS), "Number of user goals should match!"
    user_goals_len = len(gitlab_user_goals.GOALS)
    injection_format_list = [PromptInjectionFormat.GOAL_HIJACKING_PLAIN_TEXT,
                             PromptInjectionFormat.GOAL_HIJACKING_URL_INJECTION]
    results_dict = defaultdict(int)
    all_rows = []

    # The --model flag does not reach PTE-backed agents (react_api_web, pte,
    # beyond_browsing, claude_code); they source their LLM from PTE config/config.yaml.
    # Record the model that actually runs so the manifest and CSV don't mislabel it.
    effective_model = resolve_effective_model(output_format, model)

    # ------- run-level provenance manifest (written up front, includes the idx map)
    os.makedirs(output_dir, exist_ok=True)
    cells = {}
    for ug in range(user_goal_start, user_goals_len):
        for i, injf in enumerate(injection_format_list):
            idx = ug * len(injection_format_list) + i
            cells[str(idx)] = {"user_goal_idx": ug, "injection_format": injf}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": config,
        "config_stem": os.path.splitext(os.path.basename(config))[0],
        "model": effective_model,        # backbone that actually runs the agent
        "requested_model": model,        # the --model flag as passed (may be ignored by PTE agents)
        "system_prompt": system_prompt,
        "output_format": output_format,
        "output_dir": os.path.abspath(output_dir),
        "user_goal_start": user_goal_start,
        "num_user_goals": user_goals_len,
        "injection_formats": injection_format_list,
        "multi_docker": multi_docker,
        "run_single": run_single,
        "cells": cells,  # idx -> {user_goal_idx, injection_format}
    }
    with open(os.path.join(output_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
    # -------

    for user_goal_idx in range(user_goal_start, user_goals_len):
        print(f"$$$$$$$ Running {user_goal_idx+1} our of {user_goals_len} user goals, current one: "
              f"(gitlab) '{gitlab_user_goals.GOALS[user_goal_idx]}', "
              f"(reddit) '{reddit_user_goals.GOALS[user_goal_idx]}'")
        for i, injection_format in enumerate(injection_format_list):
            print(f"$$$$$$$ Running {i+1} out of {len(injection_format_list)} injection formats, current one: {injection_format}")

            rows = run_single_end_to_end(config=config,
                                         model=model,
                                         system_prompt=system_prompt,
                                         output_dir=output_dir,
                                         output_format=output_format,
                                         user_goal_idx=user_goal_idx,
                                         injection_format=injection_format,
                                         results_dict=results_dict,
                                         output_dir_idx=user_goal_idx * len(injection_format_list) + i,
                                         multi_docker=multi_docker)
            all_rows.extend(rows)

            # Persist aggregates after every cell so a run interrupted midway is still
            # partially retraceable.
            _write_run_summary(output_dir, results_dict, all_rows)

            print(f"\nAccumulated results after user_goal_idx = {user_goal_idx+1} and injection_format_idx = {i+1}: ")
            for key, value in results_dict.items():
                print(f"{key} = {value}")

            if run_single:
                print("\n!!! Running a single user goal and a single injection format is requested. Terminating")
                return

    print("\n\nDone running all experiments! Final results:")
    for key, value in results_dict.items():
        print(f"{key} = {value}")
    print(f"\nWrote per-template results ({len(all_rows)} rows) to {os.path.join(output_dir, 'results_all.csv')}")
    print(f"Wrote summary to {os.path.join(output_dir, 'summary.json')}")


def _write_run_summary(output_dir, results_dict, all_rows):
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(dict(results_dict), f, indent=4)
    if all_rows:
        with open(os.path.join(output_dir, "results_all.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=collect_results._COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)


@click.command()
@click.option(
    "--config",
    type=str,
    default="configs/experiment_config.raw.json",
    help="Where to find the config for prompt injections",
)
@click.option(
    "--model",
    type=click.Choice(['gpt-4o', 'gpt-4o-mini', 'claude-35', 'claude-37'], case_sensitive=False),
    default="gpt-4o",
    help="backbone LLM. Available options: gpt-4o, gpt-4o-mini, claude-35, claude-37",
)
@click.option(
    "--system-prompt",
    type=str,
    default="configs/system_prompts/wa_p_som_cot_id_actree_3s.json",
    help="system_prompt for the backbone LLM. Default = VWA's SOM system prompt for GPT scaffolding",
)
@click.option(
    "--output-dir",
    type=str,
    default="/tmp/computer-use-agent-logs",
    help="Folder to store the output configs and commands to run the agent",
)
@click.option(
    "--output-format",
    type=str,
    default="webarena",
    help="Format of the agentic scaffolding: webarena (default), claude, gpt_web_tools, pte, beyond_browsing, react_api_web, claude_code",
)
@click.option(
    "--run-single",
    is_flag=True,
    default=False,
    help="whether to test only a single user goal and a single injection format",
)
@click.option(
    "--user_goal_start",
    type=int,
    default=0,
    help="starting user_goal index (between 0 and total number of benign user goals)",
)
@click.option(
    "--multi-docker",
    is_flag=True,
    default=False,
    help="Use the multi-docker worker pool for parallel task execution (pte and beyond_browsing formats only)",
)
def main(config,
         model,
         system_prompt,
         output_dir,
         output_format,
         run_single,
         user_goal_start,
         multi_docker):
    print("Arguments provided to run.py: \n", locals(), "\n\n")
    run_all(config=config,
            model=model,
            system_prompt=system_prompt,
            output_dir=output_dir,
            output_format=output_format,
            run_single=run_single,
            user_goal_start=user_goal_start,
            multi_docker=multi_docker)


if __name__ == '__main__':
    main()
