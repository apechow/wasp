# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Wrapper that runs one WASP task through the ReAct API+Web agent (CodeActAgent)
and writes a gpt_web_tools-compatible JSONL trace for the WASP evaluators.

Called once per task by the generated run_agent.sh script (single-docker mode).
In multi-docker mode, run_task_multi_docker.py handles the full lifecycle.
"""

import asyncio
import json
import os
import sys
import time
import traceback
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


def _resolve_gitlab_token(base_url: str) -> str:
    """Return a PAT that actually authenticates against base_url.

    PTE's fallback chain (react_agent_runner.py) ends at config/.server_env,
    whose token belongs to a different GitLab snapshot than the one WASP
    provisions. That token 401s on every request, and the agent burns its
    iteration budget rediscovering session-cookie auth instead of doing the
    task — so verify the token here rather than trusting the chain.
    """
    import requests

    def _works(tok: str) -> bool:
        if not tok:
            return False
        try:
            r = requests.get(
                f"{base_url}/api/v4/user",
                headers={"PRIVATE-TOKEN": tok},
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    tok = os.environ.get("GITLAB_TOKEN", "").strip()
    if _works(tok):
        return tok

    from api.gitlab_pw.tokens import get_glpat  # PTE; needs sys.path set first

    tok = get_glpat(base_url)
    if not _works(tok):
        raise RuntimeError(f"Minted GLPAT does not authenticate against {base_url}")
    return tok


def _build_jsonl_lines(intent: str, steps: list) -> list:
    system_msg = {"role": "system", "content": "ReactAPIWebAgent"}
    user_msg = {"role": "user", "content": intent}

    messages = [system_msg, user_msg]
    lines = [list(messages)]

    for step in steps:
        step_type = step.get("type", "unknown")
        if step_type == "ipython":
            name = "python"
            arguments = json.dumps({"code": step.get("code", "")})
            observation = step.get("output", "")
        elif step_type == "bash":
            name = "bash"
            arguments = json.dumps({"command": step.get("command", "")})
            observation = step.get("output", "")
        elif step_type == "browse":
            name = "browse"
            arguments = json.dumps({"browser_actions": step.get("actions", "")})
            observation = step.get("content", "")
        else:
            continue

        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": name,
                    "arguments": arguments,
                }
            }],
        })
        messages.append({"role": "tool", "content": observation})
        lines.append(list(messages))

    return lines


@click.command()
@click.option("--task-config", required=True, help="Path to WASP task JSON file")
@click.option("--trace-log-dir", required=True, help="Directory to write {task_id}.jsonl trace")
@click.option("--pte-dir", required=True, help="Path to the PTE project root")
@click.option("--max-iterations", default=30, show_default=True, help="Max ReAct loop iterations per task")
@click.option("--web-only", is_flag=True, default=False, help="Use the browser-only WebAgentRunner (no API/Python/bash channel).")
def main(task_config, trace_log_dir, pte_dir, max_iterations, web_only):
    # Resolve the model the PTE runner will actually use (from PTE config/config.yaml),
    # NOT the WASP --model flag, which never reaches this agent. Recorded in meta.json
    # below so results provenance reflects the real backbone (e.g. claude-sonnet-4-6).
    from utils import read_pte_agent_model
    agent_model = read_pte_agent_model(pte_dir)

    sys.path.insert(0, str(pte_dir))
    # Web-only mode swaps in WebAgentRunner (browser-only, no API/Python/bash channel).
    # It is a drop-in subclass of ReactAgentRunner — same __init__ signature and same
    # server/base_url/glpat/_init_agent/_run_task interface — so nothing else changes.
    if web_only:
        from web_agent.web_agent_runner import WebAgentRunner as _Runner
    else:
        from react_agent.react_agent_runner import ReactAgentRunner as _Runner

    with open(task_config) as f:
        task = json.load(f)

    start_url: str = task["start_url"]
    intent: str = task["intent"]
    task_id: int = task["task_id"]

    parsed = urlparse(start_url)
    gitlab_url = f"{parsed.scheme}://{parsed.netloc}"

    # Resolved here, in the synchronous section: get_glpat uses sync Playwright,
    # which cannot run inside the asyncio loop below. Normally the batch has
    # already exported a valid GITLAB_TOKEN (run_agent.sh resolves it once), so
    # _resolve_gitlab_token returns it via the fast path with no Playwright mint.
    # Fail open: a token failure here must never abort the batch — one task
    # scoring low is fine, killing the whole run is not. (get_glpat is still
    # attempted inside _resolve_gitlab_token as a per-task fallback.)
    try:
        gitlab_token = _resolve_gitlab_token(gitlab_url)
        print(f"[run_react_api_web_agent] GitLab token OK for {gitlab_url}", flush=True)
    except Exception:
        print(f"[run_react_api_web_agent] Task {task_id}: token pre-flight failed; "
              f"running fail-open (unauthenticated).", flush=True)
        traceback.print_exc()
        gitlab_token = os.environ.get("GITLAB_TOKEN", "").strip()
    os.environ["GITLAB_TOKEN"] = gitlab_token

    run_status = {"crashed": False, "exception": None}

    async def _run() -> list:
        runner = _Runner(
            gitlab_base_url=gitlab_url,
            max_iterations=max_iterations,
        )
        runner.server = "gitlab"
        runner.base_url = gitlab_url
        runner.glpat = gitlab_token
        print(f"[run_react_api_web_agent] Initializing agent...", flush=True)
        await runner._init_agent()
        print(f"[run_react_api_web_agent] Running task: {intent[:80]!r}", flush=True)
        try:
            result = await runner._run_task(task)
            print(f"[run_react_api_web_agent] Task complete.", flush=True)
            return result.get("execution_result", {}).get("steps", [])
        except Exception as e:
            # Salvage the partial trace rather than writing an empty one that
            # silently scores 0, and keep the batch going for the other tasks.
            print(f"[run_react_api_web_agent] Task {task_id} crashed:", flush=True)
            traceback.print_exc()
            run_status["crashed"] = True
            run_status["exception"] = f"{type(e).__name__}: {e}"
            return list(getattr(runner, "_last_steps", []))

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    steps = asyncio.run(_run())
    duration_s = round(time.time() - t0, 3)
    lines = _build_jsonl_lines(intent, steps)

    Path(trace_log_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(trace_log_dir, f"{task_id}.jsonl")
    with open(out_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    _write_meta(trace_log_dir, task_id, {
        "task_id": task_id,
        "agent": "WebAgentRunner" if web_only else "ReactAPIWebAgent",
        "web_only": web_only,
        "model": agent_model,  # actual backbone from PTE config.yaml, not the WASP --model flag
        "intent": intent,
        "start_url": start_url,
        "gitlab_url": gitlab_url,
        "max_iterations": max_iterations,
        "num_steps": len(lines) - 1,
        "crashed": run_status["crashed"],
        "exception": run_status["exception"],
        "started_at": started_at,
        "duration_s": duration_s,
    })

    print(f"[run_react_api_web_agent] Task {task_id}: wrote {len(lines) - 1} steps to {out_path}", flush=True)


if __name__ == "__main__":
    main()
