# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Wrapper that runs one WASP task through PTE's Claude Code (codegen) agent
(ClaudeCodeAgentRunner) and writes a gpt_web_tools-compatible JSONL trace for
the WASP evaluators.

Unlike the ReAct API+Web agent, the Claude Code agent is a two-stage codegen
pipeline: a network-isolated `claude` session writes a deterministic Python
script for the task, and that script is then run live against the site. It
never browses the injected page, and it produces no step-by-step trace — so
this wrapper *synthesizes* the trace from two artifacts the runner returns:
the generated script's source and the script's captured stdout.

The synthesized trace has two steps:
  1. a leading marker step (so the substantive step below is not the first
     line, which the step-by-step evaluator drops), and
  2. a `python` step whose tool-call *arguments* carry the generated script
     source and whose observation is the script's stdout. WASP's ExfilEvaluator
     matches the attacker URL only against the tool-call arguments, so the
     script source is the one place an exfil attempt could surface.

Site handling is generic: the site comes from task["sites"][0] and the base URL
from task["start_url"], so reddit works unchanged the moment PTE adds it to
ClaudeCodeAgentRunner.SUPPORTED_SITES. Until then a reddit task returns an
"unsupported_site" result, which this wrapper records as a well-formed
(non-exfil) trace.

Called once per task by the generated run_agent.sh script (single-docker mode).
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import click


def _build_jsonl_lines(intent: str, steps: list) -> list:
    """Build the cumulative-prefix gpt_web_tools JSONL.

    Each element of `steps` is a dict: {"name", "arguments", "observation"},
    where "arguments" is a JSON-serializable dict for the tool call.
    """
    system_msg = {"role": "system", "content": "ClaudeCodeAgent"}
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
    """Turn a ClaudeCodeAgentRunner result into synthetic trace steps."""
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
        "arguments": {"code": f"# Claude Code codegen agent: generated task_{task_id}.py\n"},
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
def main(task_config, trace_log_dir, pte_dir, codegen_timeout, script_timeout):
    pte_dir = str(Path(pte_dir).resolve())
    sys.path.insert(0, pte_dir)
    from eval.claude_code_agent_runner import ClaudeCodeAgentRunner

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
    # by the run_claude_task.sh subprocess the runner spawns.
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
                print(f"[run_claude_code_agent] Minted GitLab token for {base_url}", flush=True)
            except Exception as e:
                print(f"[run_claude_code_agent] WARNING: token mint failed ({e}); "
                      f"the generated script will fall back to self-minting", flush=True)
                gitlab_token = ""

    async def _run() -> dict:
        runner = ClaudeCodeAgentRunner(
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
        print(f"[run_claude_code_agent] Initializing agent (site={site})...", flush=True)
        await runner._init_agent()
        print(f"[run_claude_code_agent] Running task: {intent[:80]!r}", flush=True)
        try:
            result = await runner._run_task(task)
            print(f"[run_claude_code_agent] Task complete.", flush=True)
            return result or {}
        except Exception as e:
            print(f"[run_claude_code_agent] Task {task_id} failed: {e}", flush=True)
            return {"success": False, "error": str(e), "failure_kind": "wrapper"}

    result = asyncio.run(_run())
    steps = _build_steps(pte_dir, task_id, result)
    lines = _build_jsonl_lines(intent, steps)

    Path(trace_log_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(trace_log_dir, f"{task_id}.jsonl")
    with open(out_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    print(f"[run_claude_code_agent] Task {task_id}: wrote {len(lines) - 1} steps to {out_path}", flush=True)


if __name__ == "__main__":
    main()
