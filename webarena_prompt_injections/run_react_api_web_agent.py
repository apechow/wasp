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
import traceback
from pathlib import Path
from urllib.parse import urlparse

import click


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
def main(task_config, trace_log_dir, pte_dir, max_iterations):
    sys.path.insert(0, str(pte_dir))
    from react_agent.react_agent_runner import ReactAgentRunner

    with open(task_config) as f:
        task = json.load(f)

    start_url: str = task["start_url"]
    intent: str = task["intent"]
    task_id: int = task["task_id"]

    parsed = urlparse(start_url)
    gitlab_url = f"{parsed.scheme}://{parsed.netloc}"

    # Minted here, in the synchronous section: get_glpat uses sync Playwright,
    # which cannot run inside the asyncio loop below. Raising aborts the batch
    # (run_agent.sh uses `set -e`) — correct, since without a working token
    # every task in it fails anyway.
    gitlab_token = _resolve_gitlab_token(gitlab_url)
    os.environ["GITLAB_TOKEN"] = gitlab_token
    print(f"[run_react_api_web_agent] GitLab token OK for {gitlab_url}", flush=True)

    async def _run() -> list:
        runner = ReactAgentRunner(
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
        except Exception:
            # Salvage the partial trace rather than writing an empty one that
            # silently scores 0, and keep the batch going for the other tasks.
            print(f"[run_react_api_web_agent] Task {task_id} crashed:", flush=True)
            traceback.print_exc()
            return list(getattr(runner, "_last_steps", []))

    steps = asyncio.run(_run())
    lines = _build_jsonl_lines(intent, steps)

    Path(trace_log_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(trace_log_dir, f"{task_id}.jsonl")
    with open(out_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    print(f"[run_react_api_web_agent] Task {task_id}: wrote {len(lines) - 1} steps to {out_path}", flush=True)


if __name__ == "__main__":
    main()
