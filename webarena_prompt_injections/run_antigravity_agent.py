# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Wrapper that runs one WASP task through PTE's Antigravity (Gemini "agy" CLI)
codegen agent (AntigravityAgentRunner) and writes a gpt_web_tools-compatible
JSONL trace for the WASP evaluators.

This is the antigravity sibling of run_claude_code_agent.py: same two-stage
codegen pipeline, same synthesized trace, only the PTE runner class differs.
A network-isolated `agy` session writes a deterministic Python script for the
task, and that script is then run live against the site. It never browses the
injected page, so this wrapper *synthesizes* the trace from two artifacts the
runner returns: the generated script's source and the script's captured stdout.

The synthesized trace has two steps:
  1. a leading marker step (so the substantive step below is not the first
     line, which the step-by-step evaluator drops), and
  2. a `python` step whose tool-call *arguments* carry the generated script
     source and whose observation is the script's stdout. WASP's ExfilEvaluator
     matches the attacker URL only against the tool-call arguments, so the
     script source is the one place an exfil attempt could surface.

Site handling is generic: the site comes from task["sites"][0] and the base URL
from task["start_url"]. gitlab, reddit, shopping and shopping_admin are all in
AntigravityAgentRunner.SUPPORTED_SITES; any other site returns an
"unsupported_site" result, which this wrapper records as a well-formed
(non-exfil) trace.

Config gate: the PTE AntigravityAgentRunner drives run_antigravity_task.sh,
which hard-errors unless PTE config/config.yaml has agent_llm_provider set to
`google` or `antigravity`. This wrapper does not flip that file — set the
provider (and an agent_llm_model that `agy models` offers) before running.

Called once per task by the generated run_agent.sh script (single-docker mode).
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import click


def _write_meta(trace_log_dir: str, task_id: int, meta: dict) -> None:
    """Persist per-task provenance next to the {task_id}.jsonl trace so a short/crashed
    trace is distinguishable from a successful one and the run stays retraceable."""
    Path(trace_log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(trace_log_dir, f"{task_id}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _build_jsonl_lines(intent: str, steps: list) -> list:
    """Build the cumulative-prefix gpt_web_tools JSONL.

    Each element of `steps` is a dict: {"name", "arguments", "observation"},
    where "arguments" is a JSON-serializable dict for the tool call.
    """
    system_msg = {"role": "system", "content": "AntigravityAgent"}
    user_msg = {"role": "user", "content": intent}

    messages = [system_msg, user_msg]
    lines = [list(messages)]

    for step in steps:
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": step["name"],
                    "arguments": json.dumps(step["arguments"]),
                }
            }],
        })
        messages.append({"role": "tool", "content": step["observation"]})
        lines.append(list(messages))

    return lines


def _read_script_source(pte_dir: str, result: dict) -> str:
    """Return the generated script's source, or "" if it can't be located.

    `result["script"]` is a path relative to the PTE repo root (set on both the
    happy path and the script_execution failure path).
    """
    rel = result.get("script")
    if not rel:
        return ""
    path = os.path.join(pte_dir, rel)
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _build_steps(pte_dir: str, task_id: int, result: dict) -> list:
    """Turn an AntigravityAgentRunner result into synthetic trace steps."""
    answer = result.get("answer")
    error = result.get("error")
    failure_kind = result.get("failure_kind")
    script_output = result.get("script_output") or ""
    script_source = _read_script_source(pte_dir, result)

    # Leading marker step. Keeps the substantive script step off line 1 (which
    # the step-by-step evaluator drops), and records the codegen outcome.
    if failure_kind:
        marker_obs = f"codegen/execution failed (failure_kind={failure_kind}): {error}"
    else:
        marker_obs = f"Generated implementation script for task {task_id}; answer: {answer!r}"
    steps = [{
        "name": "python",
        "arguments": {"code": f"# Antigravity codegen agent: generated task_{task_id}.py\n"},
        "observation": marker_obs,
    }]

    # Substantive step: the generated script's source as the tool-call
    # arguments, its live stdout as the observation. Present whenever a script
    # exists (including a script that failed its live run).
    if script_source:
        steps.append({
            "name": "python",
            "arguments": {"code": script_source},
            "observation": script_output or (error or ""),
        })

    return steps


@click.command()
@click.option("--task-config", required=True, help="Path to WASP task JSON file")
@click.option("--trace-log-dir", required=True, help="Directory to write {task_id}.jsonl trace")
@click.option("--pte-dir", required=True, help="Path to the PTE project root")
@click.option("--codegen-timeout", default=900, show_default=True, help="Max seconds for the codegen session")
@click.option("--script-timeout", default=120, show_default=True, help="Max seconds for the live script run")
@click.option("--max-attempts", default=3, show_default=True, help="Max codegen/execution attempts per task; a failed live run is retried in replan mode, a failed codegen is regenerated. 1 disables retries.")
def main(task_config, trace_log_dir, pte_dir, codegen_timeout, script_timeout, max_attempts):
    max_attempts = max(1, max_attempts)
    pte_dir = str(Path(pte_dir).resolve())
    sys.path.insert(0, pte_dir)
    from eval.antigravity_agent_runner import AntigravityAgentRunner

    with open(task_config) as f:
        task = json.load(f)

    start_url: str = task["start_url"]
    intent: str = task["intent"]
    task_id: int = task["task_id"]
    sites = task.get("sites") or []
    site = (sites[0] if sites else "gitlab").lower()

    parsed = urlparse(start_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    # The codegen session resolves the task prompt via PTE/scripts/lookup_task.py,
    # which keys on PTE's own dataset. WASP task ids/intents aren't in it, so hand
    # the intent (and the project path) through the environment; lookup_task.py
    # honors PTE_TASK_INTENT and skips the dataset search when it is set. Inherited
    # by the run_antigravity_task.sh subprocess the runner spawns.
    os.environ["PTE_TASK_INTENT"] = intent
    os.environ["PTE_TASK_PROJECT_PATH"] = parsed.path or ""

    # Provision a GitLab API token for the generated script. Without one, the
    # script falls back to `config.init_tokens.refresh_gitlab_token`, an import
    # some codegen sessions omit the PTE-root sys.path setup for (and it also
    # drives a browser login per script) — the dominant cause of task failures.
    # Supplying GITLAB_TOKEN up front (the codegen contract's first choice) skips
    # that path entirely. Reuse a caller-exported token if present (one token for
    # the whole run, no per-task login); otherwise mint one once for this process.
    # Minted here, in the synchronous section: get_glpat uses sync Playwright,
    # which cannot run inside the asyncio loop below.
    gitlab_token = ""
    if site == "gitlab":
        gitlab_token = os.environ.get("GITLAB_TOKEN", "").strip()
        if not gitlab_token:
            try:
                from api.gitlab_pw.tokens import get_glpat
                gitlab_token = get_glpat(base_url)
                print(f"[run_antigravity_agent] Minted GitLab token for {base_url}", flush=True)
            except Exception as e:
                print(f"[run_antigravity_agent] WARNING: token mint failed ({e}); "
                      f"the generated script will fall back to self-minting", flush=True)
                gitlab_token = ""

    run_status = {"attempts": 0}

    async def _run() -> dict:
        runner = AntigravityAgentRunner(
            gitlab_base_url=base_url,
            regenerate_code=True,   # WASP ids aren't cached; always codegen fresh
            enable_reset=False,     # never wipe the injected environment
            codegen_timeout=codegen_timeout,
            script_timeout=script_timeout,
        )
        runner.server = site
        runner.base_url = base_url
        if gitlab_token:
            runner.site_glpat = gitlab_token
        print(f"[run_antigravity_agent] Initializing agent (site={site})...", flush=True)
        await runner._init_agent()
        print(f"[run_antigravity_agent] Running task: {intent[:80]!r}", flush=True)

        # Attempt loop mirroring PTE's pytest harness
        # (eval/tests/test_agent_verified_all_sites.py). The wrapper has no
        # grader, so "done" means the runner produced an answer (no
        # failure_kind). A script_execution failure is retried in replan mode
        # (the runner reads the failed script's own python_output.log and
        # repairs it); a sandbox_write_block failure is regenerated WITH a
        # corrective hint (there is no script to repair — the session died at
        # the sandbox write-boundary); a codegen failure is regenerated fresh.
        attempt = 1
        result: dict = {}
        while True:
            # Set from the *previous* result: attempt 1 always runs fresh; a
            # retry after a live-run failure replans, a retry after the sandbox
            # write-boundary regenerates with a correction, a retry after a
            # codegen failure regenerates plain.
            prev_kind = result.get("failure_kind")
            runner.replan = attempt > 1 and prev_kind == "script_execution"
            runner.codegen_correction = (
                "sandbox_write_block"
                if attempt > 1 and prev_kind == "sandbox_write_block"
                else None
            )
            runner.replan_attempt = attempt
            runner.replan_max_attempts = max_attempts
            try:
                result = await runner._run_task(task) or {}
            except Exception as e:
                print(f"[run_antigravity_agent] Task {task_id} attempt {attempt} failed: {e}", flush=True)
                result = {"success": False, "error": str(e), "failure_kind": "wrapper"}

            failure_kind = result.get("failure_kind")
            # Got an answer (no failure_kind) => done. Never retry a produced
            # answer: WASP grades separately and retrying would only fit its
            # grader.
            if not failure_kind or attempt >= max_attempts:
                break
            # Only the runner's own retryable kinds are worth another attempt;
            # unsupported_site and the wrapper `except` kind are terminal.
            if failure_kind not in ("script_execution", "sandbox_write_block", "codegen"):
                break

            attempt += 1
            if failure_kind == "script_execution":
                print(f"[run_antigravity_agent] Task {task_id} attempt {attempt}/{max_attempts} "
                      f"script failed ({result.get('error')}) — replanning from its logs", flush=True)
            elif failure_kind == "sandbox_write_block":
                print(f"[run_antigravity_agent] Task {task_id} attempt {attempt}/{max_attempts} "
                      f"hit the sandbox write-boundary ({result.get('error')}) — regenerating with corrective guidance", flush=True)
            else:  # codegen
                print(f"[run_antigravity_agent] Task {task_id} attempt {attempt}/{max_attempts} "
                      f"codegen produced nothing ({result.get('error')}) — regenerating", flush=True)

        print(f"[run_antigravity_agent] Task complete (attempts={attempt}).", flush=True)
        run_status["attempts"] = attempt
        return result

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    result = asyncio.run(_run())
    duration_s = round(time.time() - t0, 3)
    steps = _build_steps(pte_dir, task_id, result)
    lines = _build_jsonl_lines(intent, steps)

    Path(trace_log_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(trace_log_dir, f"{task_id}.jsonl")
    with open(out_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    # A produced answer has no failure_kind; anything else (script_execution,
    # sandbox_write_block, codegen, wrapper, unsupported_site) means the agent
    # did not complete cleanly.
    failure_kind = result.get("failure_kind")
    _write_meta(trace_log_dir, task_id, {
        "task_id": task_id,
        "agent": "AntigravityAgent",
        "intent": intent,
        "start_url": start_url,
        "site": site,
        "num_steps": len(lines) - 1,
        "attempts": run_status["attempts"],
        "max_attempts": max_attempts,
        "codegen_timeout": codegen_timeout,
        "script_timeout": script_timeout,
        "crashed": bool(failure_kind),
        "failure_kind": failure_kind,
        "exception": result.get("error"),
        "started_at": started_at,
        "duration_s": duration_s,
    })

    print(f"[run_antigravity_agent] Task {task_id}: wrote {len(lines) - 1} steps to {out_path}", flush=True)


if __name__ == "__main__":
    main()
