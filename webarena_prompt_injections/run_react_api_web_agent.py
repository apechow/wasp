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
from pathlib import Path
from urllib.parse import urlparse

import click


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

    async def _run() -> list:
        runner = ReactAgentRunner(
            gitlab_base_url=gitlab_url,
            max_iterations=max_iterations,
        )
        runner.server = "gitlab"
        runner.base_url = gitlab_url
        print(f"[run_react_api_web_agent] Initializing agent...", flush=True)
        await runner._init_agent()
        print(f"[run_react_api_web_agent] Running task: {intent[:80]!r}", flush=True)
        try:
            result = await runner._run_task(task)
            print(f"[run_react_api_web_agent] Task complete.", flush=True)
            return result.get("execution_result", {}).get("steps", [])
        except Exception as e:
            print(f"[run_react_api_web_agent] Task {task_id} failed: {e}", flush=True)
            return []

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
