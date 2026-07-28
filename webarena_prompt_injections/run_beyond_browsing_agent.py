# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Wrapper that runs one WASP task through the Beyond Browsing (OpenDevin CodeAct)
agent and writes a gpt_web_tools-compatible JSONL trace for the WASP evaluators.

Called once per task by the generated run_agent.sh script (single-docker mode).
In multi-docker mode, run_task_multi_docker.py handles the full lifecycle.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import click


def _action_name(action) -> str:
    class_name = type(action).__name__
    mapping = {
        "CmdRunAction": "bash",
        "IPythonRunCellAction": "python",
        "BrowseInteractiveAction": "browse",
        "MessageAction": "message",
    }
    return mapping.get(class_name, class_name.lower())


def _action_args(action) -> dict:
    class_name = type(action).__name__
    if class_name == "CmdRunAction":
        return {"command": getattr(action, "command", "")}
    if class_name == "IPythonRunCellAction":
        return {"code": getattr(action, "code", "")}
    if class_name == "BrowseInteractiveAction":
        return {"browser_actions": getattr(action, "browser_actions", "")}
    if class_name == "MessageAction":
        return {"content": getattr(action, "content", "")}
    return {}


def _is_terminal_action(action) -> bool:
    return type(action).__name__ in ("AgentFinishAction", "AgentRejectAction")


def _build_jsonl_lines(intent: str, history: list) -> list:
    system_msg = {"role": "system", "content": "BeyondBrowsing Agent"}
    user_msg = {"role": "user", "content": intent}

    messages = [system_msg, user_msg]
    lines = [list(messages)]

    for act, obs in history:
        if _is_terminal_action(act):
            continue

        observation = str(obs) if not isinstance(obs, str) else obs

        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": _action_name(act),
                    "arguments": json.dumps(_action_args(act)),
                }
            }],
        })
        messages.append({"role": "tool", "content": observation})
        lines.append(list(messages))

    return lines


@click.command()
@click.option("--task-config", required=True, help="Path to WASP task JSON file")
@click.option("--trace-log-dir", required=True, help="Directory to write {task_id}.jsonl trace")
@click.option("--beyond-browsing-dir", required=True,
              help="Path to the Beyond Browsing project root (API-Based-Agent/)")
def main(task_config, trace_log_dir, beyond_browsing_dir):
    sys.path.insert(0, str(beyond_browsing_dir))
    from opendevin.core.main import main as bb_main  # noqa: E402

    with open(task_config) as f:
        task = json.load(f)

    start_url: str = task["start_url"]
    intent: str = task["intent"]
    task_id: int = task["task_id"]

    task_str = f"Go to {start_url} and {intent}"

    try:
        state = asyncio.run(bb_main(
            task_str=task_str,
            exit_on_message=True,
            fake_user_response_fn=lambda _: "continue",
        ))
        history = state.history if state is not None else []
        print(f"[run_beyond_browsing_agent] Task complete.", flush=True)
    except Exception as e:
        print(f"[run_beyond_browsing_agent] Task {task_id} failed: {e}", flush=True)
        history = []

    lines = _build_jsonl_lines(intent, history)

    Path(trace_log_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(trace_log_dir, f"{task_id}.jsonl")
    with open(out_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    print(f"[run_beyond_browsing_agent] Task {task_id}: wrote {len(lines) - 1} steps to {out_path}", flush=True)


if __name__ == "__main__":
    main()
