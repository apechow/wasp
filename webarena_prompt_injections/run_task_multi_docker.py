# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Per-task script for multi-docker WASP evaluation.

Handles the full lifecycle for one task:
  acquire worker → setup + inject (into this worker) → run agent → save worker_id → release (read_only)

Called once per task in parallel by the generated run_agent.sh.
The pi_context JSON is written by prompt_injector.py during Step 1.
"""

import asyncio
import copy
import json
import os
import sys
from pathlib import Path

import click

# Ensure WASP modules are importable regardless of cwd.
_wasp_dir = os.path.dirname(os.path.abspath(__file__))
if _wasp_dir not in sys.path:
    sys.path.insert(0, _wasp_dir)


@click.command()
@click.option("--pi-context-path", required=True,
              help="Path to pi_task_contexts/{task_id}.json written by prompt_injector.py")
@click.option("--output-format", required=True, type=click.Choice(["pte", "beyond_browsing"]),
              help="Agent scaffolding to use")
@click.option("--trace-log-dir", required=True,
              help="Directory to write {task_id}.jsonl trace")
@click.option("--pte-dir", default=None,
              help="Path to PTE project root (required for --output-format pte)")
@click.option("--beyond-browsing-dir", default=None,
              help="Path to Beyond Browsing project root (required for --output-format beyond_browsing)")
def main(pi_context_path, output_format, trace_log_dir, pte_dir, beyond_browsing_dir):
    with open(pi_context_path) as f:
        ctx = json.load(f)

    pi_config = ctx["pi_config"]
    user_goal_idx = ctx["user_goal_idx"]
    injection_format = ctx["injection_format"]
    task_id = ctx["task_id"]
    task_dir = ctx["task_dir"]
    attacker_task_dir = ctx["attacker_task_dir"]

    if output_format == "pte":
        if not pte_dir:
            raise click.UsageError("--pte-dir is required when --output-format is pte")
        asyncio.run(_run_pte(
            pi_config=pi_config,
            user_goal_idx=user_goal_idx,
            injection_format=injection_format,
            task_id=task_id,
            task_dir=task_dir,
            attacker_task_dir=attacker_task_dir,
            trace_log_dir=trace_log_dir,
            pte_dir=pte_dir,
        ))
    else:
        if not beyond_browsing_dir:
            raise click.UsageError("--beyond-browsing-dir is required when --output-format is beyond_browsing")
        _run_beyond_browsing(
            pi_config=pi_config,
            user_goal_idx=user_goal_idx,
            injection_format=injection_format,
            task_id=task_id,
            task_dir=task_dir,
            attacker_task_dir=attacker_task_dir,
            trace_log_dir=trace_log_dir,
            beyond_browsing_dir=beyond_browsing_dir,
        )


async def _run_pte(
    pi_config, user_goal_idx, injection_format, task_id,
    task_dir, attacker_task_dir, trace_log_dir, pte_dir,
):
    if str(pte_dir) not in sys.path:
        sys.path.append(str(pte_dir))
    from eval.docker.workers_new import worker_session  # noqa: E402

    from environment_editors.gitlab_editor import GitlabEditor
    from environment_editors.reddit_editor import RedditEditor
    from prompt_injector import WebArenaPromptInjector

    environment = pi_config["environment"]

    async with worker_session(str(task_id), server=environment, read_only=True) as w:
        worker_url = w["server_url"]
        worker_id = w["worker_id"]

        if environment == "gitlab":
            editor = GitlabEditor(worker_url)
        elif environment == "reddit":
            editor = RedditEditor(worker_url)
        else:
            raise ValueError(f"Unknown environment: {environment!r}")

        injector = WebArenaPromptInjector([editor], [copy.deepcopy(pi_config)])

        # Playwright-based injection must run in a thread (it's synchronous).
        task_dict, attacker_task_dict = await asyncio.to_thread(
            injector.setup_and_inject_single_config,
            injector.prompt_injection_configs[0],
            editor,
            user_goal_idx,
            injection_format,
            task_id,
        )

        _write_task_configs(task_dict, attacker_task_dict, task_dir, attacker_task_dir, task_id)
        _save_worker_id(worker_id, trace_log_dir, task_id)

        start_url = task_dict["start_url"]
        intent = task_dict["intent"]

        from run_pte_agent import _run_task_with_servers  # noqa: E402
        await _run_task_with_servers(task_id, intent, start_url, worker_url, pte_dir, trace_log_dir)
        # worker_session releases with read_only=True on context exit


def _run_beyond_browsing(
    pi_config, user_goal_idx, injection_format, task_id,
    task_dir, attacker_task_dir, trace_log_dir, beyond_browsing_dir,
):
    bb_eval_dir = os.path.join(beyond_browsing_dir, "evaluation", "webarena")
    # Append (not insert) so WASP's _wasp_dir stays first: prevents BB's evaluation/webarena/utils.py
    # from shadowing WASP's utils.py when prompt_injector.py does 'from utils import ...'.
    if bb_eval_dir not in sys.path:
        sys.path.append(bb_eval_dir)
    if str(beyond_browsing_dir) not in sys.path:
        sys.path.append(str(beyond_browsing_dir))

    from worker_pool.workers import acquire_worker, release_worker, server_urls_for_worker  # noqa: E402
    from opendevin.core.main import main as bb_main  # noqa: E402

    from environment_editors.gitlab_editor import GitlabEditor
    from environment_editors.reddit_editor import RedditEditor
    from prompt_injector import WebArenaPromptInjector

    environment = pi_config["environment"]

    worker = acquire_worker(str(task_id))
    worker_id = worker["worker_id"]
    urls = server_urls_for_worker(worker)

    try:
        if environment == "gitlab":
            worker_url = urls.get("GITLAB", "")
            editor = GitlabEditor(worker_url)
        elif environment == "reddit":
            worker_url = urls.get("REDDIT", "")
            editor = RedditEditor(worker_url)
        else:
            raise ValueError(f"Unknown environment: {environment!r}")

        injector = WebArenaPromptInjector([editor], [copy.deepcopy(pi_config)])

        task_dict, attacker_task_dict = injector.setup_and_inject_single_config(
            injector.prompt_injection_configs[0],
            editor,
            user_goal_idx,
            injection_format,
            task_id,
        )

        _write_task_configs(task_dict, attacker_task_dict, task_dir, attacker_task_dir, task_id)
        _save_worker_id(worker_id, trace_log_dir, task_id)

        start_url = task_dict["start_url"]
        intent = task_dict["intent"]
        task_str = f"Go to {start_url} and {intent}"

        orig_env = {k: os.environ.get(k) for k in urls}
        os.environ.update(urls)

        try:
            state = asyncio.run(bb_main(
                task_str=task_str,
                exit_on_message=True,
                fake_user_response_fn=lambda _: "continue",
            ))
            history = state.history if state is not None else []
            print(f"[run_task_multi_docker] Task {task_id} complete.", flush=True)
        except Exception as e:
            print(f"[run_task_multi_docker] Task {task_id} failed: {e}", flush=True)
            history = []
        finally:
            for k, v in orig_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        from run_beyond_browsing_agent import _build_jsonl_lines  # noqa: E402
        _write_trace(intent, history, trace_log_dir, task_id, _build_jsonl_lines)

    finally:
        release_worker(worker_id, read_only=True)


def _write_task_configs(task_dict, attacker_task_dict, task_dir, attacker_task_dir, task_id):
    Path(task_dir).mkdir(parents=True, exist_ok=True)
    Path(attacker_task_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(task_dir, f"{task_id}.json"), "w") as f:
        json.dump(task_dict, f, indent=4)
    with open(os.path.join(attacker_task_dir, f"{task_id}.json"), "w") as f:
        json.dump(attacker_task_dict, f, indent=4)


def _save_worker_id(worker_id, trace_log_dir, task_id):
    worker_ids_dir = os.path.join(os.path.dirname(os.path.abspath(trace_log_dir)), "worker_ids")
    os.makedirs(worker_ids_dir, exist_ok=True)
    with open(os.path.join(worker_ids_dir, f"{task_id}.txt"), "w") as f:
        f.write(str(worker_id))


def _write_trace(intent, history, trace_log_dir, task_id, build_fn):
    lines = build_fn(intent, history)
    Path(trace_log_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(trace_log_dir, f"{task_id}.jsonl")
    with open(out_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    print(f"[run_task_multi_docker] Task {task_id}: wrote {len(lines) - 1} steps to {out_path}", flush=True)


if __name__ == "__main__":
    main()
