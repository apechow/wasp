# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Wrapper that runs one WASP task through the PTE agent and writes a
gpt_web_tools-compatible JSONL trace for the WASP evaluators.

Called once per task by the generated run_agent.sh script (single-docker mode).
In multi-docker mode, run_task_multi_docker.py calls _run_task_with_servers directly.
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


def _tool_name(step) -> str:
    tn = getattr(step, "tool_name", None)
    if tn is None:
        return "unknown"
    return tn.value if hasattr(tn, "value") else str(tn)


def _step_args(step) -> dict:
    args = {}
    for arg in getattr(step, "arguments", []) or []:
        args[arg.name] = arg.value
    return args


def _build_jsonl_lines(intent: str, plan, outputs: dict) -> list:
    system_msg = {"role": "system", "content": "PTE Agent"}
    user_msg = {"role": "user", "content": intent}

    messages = [system_msg, user_msg]
    lines = [list(messages)]

    if plan is None:
        return lines

    for step in plan.plan:
        if getattr(step, "step_type", "tool_call") == "conditional":
            continue

        sid = step.step_id
        raw = outputs.get(sid, {})
        observation = json.dumps(raw) if not isinstance(raw, str) else raw

        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": _tool_name(step),
                    "arguments": json.dumps(_step_args(step)),
                }
            }],
        })
        messages.append({"role": "tool", "content": observation})
        lines.append(list(messages))

    return lines


async def _run_task_with_servers(
    task_id: int,
    intent: str,
    start_url: str,
    gitlab_host: str,
    pte_dir: str,
    trace_log_dir: str,
):
    from agent.agent import Agent

    servers = {
        "gitlab": gitlab_host,
        "reddit": "http://127.0.0.1:7791",
    }
    agent = Agent(
        env_file=os.path.join(pte_dir, "config", ".server_env"),
        api_dir=os.path.join(pte_dir, "api"),
    )
    print(f"[run_pte_agent] Initializing agent...", flush=True)
    agent.initialize(servers)
    print(f"[run_pte_agent] Running task: {intent[:80]!r}", flush=True)

    crashed = False
    exception = None
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    try:
        result = await agent.run_task(
            f"Go to {start_url} and {intent}",
            servers=servers,
        )
        print(f"[run_pte_agent] Task complete.", flush=True)
        plan = agent.last_plan_response
        outputs = result.outputs
    except Exception as e:
        print(f"[run_pte_agent] Task {task_id} failed: {e}", flush=True)
        crashed = True
        exception = f"{type(e).__name__}: {e}"
        plan = None
        outputs = {}
    duration_s = round(time.time() - t0, 3)

    lines = _build_jsonl_lines(intent, plan, outputs)

    Path(trace_log_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(trace_log_dir, f"{task_id}.jsonl")
    with open(out_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    _write_meta(trace_log_dir, task_id, {
        "task_id": task_id,
        "agent": "PTE Agent",
        "intent": intent,
        "start_url": start_url,
        "num_steps": len(lines) - 1,
        "crashed": crashed,
        "exception": exception,
        "started_at": started_at,
        "duration_s": duration_s,
    })

    print(f"[run_pte_agent] Task {task_id}: wrote {len(lines) - 1} steps to {out_path}", flush=True)


@click.command()
@click.option("--task-config", required=True, help="Path to WASP task JSON file")
@click.option("--trace-log-dir", required=True, help="Directory to write {task_id}.jsonl trace")
@click.option("--pte-dir", required=True, help="Path to the PTE project root")
def main(task_config, trace_log_dir, pte_dir):
    sys.path.insert(0, str(pte_dir))

    with open(task_config) as f:
        task = json.load(f)

    start_url: str = task["start_url"]
    intent: str = task["intent"]
    task_id: int = task["task_id"]

    parsed = urlparse(start_url)
    gitlab_host = f"{parsed.scheme}://{parsed.netloc}"

    asyncio.run(_run_task_with_servers(task_id, intent, start_url, gitlab_host, pte_dir, trace_log_dir))


if __name__ == "__main__":
    main()
