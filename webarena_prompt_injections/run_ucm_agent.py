# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Wrapper that runs one WASP task through PTE's trusted UCM agent
(screenshot-driven Claude computer-use with untrusted-content masking +
quarantined LLM) and writes a gpt_web_tools-compatible JSONL trace for the WASP
evaluators.

Called once per task by the generated run_agent.sh script (single-docker mode).

Settings mirror PTE/eval/tests/test_ucm_agent_gitlab.py: masking ON
(reveal_mode="trusted") and the quarantined LLM ON (system_prompt_name=
"ucm_defense") — the UcmAgentRunner defaults. WASP owns environment setup /
injection and grading, so we run the agent with enable_reset=False and call
runner._run_task directly (NOT run_agent_on_task, which would add PTE's own
GitLab reset + grading).

The UCM agent is GitLab-only (UcmAgentRunner.SUPPORTED_SITES == {"gitlab"});
reddit (or any other) tasks get a graceful system+user-only trace and a meta
note rather than crashing the batch.

Offline smoke test (no LLM, no docker): set UCM_DRY_RUN=1 (or pass
--dry-run-responses <path>) to skip the runner entirely and build the trace +
meta from a fixture model_responses.jsonl.
"""

import asyncio
import json
import os
import re
import subprocess
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


def _iter_jsonl(path):
    """Yield parsed objects from a JSONL file, skipping blank / malformed lines."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def extract_answer_from_entries(entries: list) -> str:
    """Mirror UcmAgentRunner.extract_agent_answer, but over already-parsed entries.

    Takes the last assistant_response with task_complete=True (else the last
    assistant_response), then strips everything from the case-insensitive
    "task complete" marker onward. Used for the trailing final_answer trace step
    (and in dry-run, where the runner did not compute the answer for us).
    """
    last_text = ""
    task_complete_text = ""
    for e in entries:
        if e.get("type") == "assistant_response" and isinstance(e.get("text"), str):
            last_text = e["text"]
            if e.get("task_complete") is True:
                task_complete_text = e["text"]
    answer = task_complete_text or last_text
    parts = re.split(r"task\s+complete", answer, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip()


def build_trace_lines(intent: str, entries: list, answer: str = None,
                      agent_label: str = "UcmAgent") -> list:
    """Convert parsed model_responses.jsonl `entries` into the cumulative-prefix
    gpt_web_tools JSONL (a list of message-list snapshots).

    Each UCM `action` entry becomes one assistant tool_call whose `name` is the
    action_type (click / type / goto / keypress / quarantined_llm_analysis / ...)
    and whose `arguments` is the JSON-dumped action dict — so exfil-relevant text
    (typed text, a goto URL) lands inside tool_calls[0].function.arguments, which
    is exactly what the WASP evaluators match against. The `tool` observation for
    each action is that step's assistant_response text (the agent's reasoning) if
    present, else a screenshot placeholder (the UCM agent's real observation is a
    screenshot, which carries no text).

    A trailing `final_answer` step carries the agent's final answer so an
    exfil-via-answer attack is also matchable. No leading marker step is needed:
    both WASP parsers (evaluator_step_by_step.create_agent_actions_based_on_gpt_tool_use_log
    and visualwebarena light_evaluation) process every real action and only drop
    the leading [system, user] line.
    """
    system_msg = {"role": "system", "content": agent_label}
    user_msg = {"role": "user", "content": intent}

    messages = [system_msg, user_msg]
    lines = [list(messages)]

    # Per-step reasoning text (assistant_response), used as the observation shown
    # alongside that step's action.
    step_text = {}
    for e in entries:
        if e.get("type") == "assistant_response" and isinstance(e.get("text"), str):
            step_text[e.get("step")] = e["text"]

    for e in entries:
        if e.get("type") != "action":
            continue
        action_type = e.get("action_type", "unknown")
        action = e.get("action", {})
        observation = step_text.get(e.get("step")) or "[screenshot]"
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": action_type,
                    "arguments": json.dumps(action),
                }
            }],
        })
        messages.append({"role": "tool", "content": observation})
        lines.append(list(messages))

    if answer:
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": "final_answer",
                    "arguments": json.dumps({"answer": answer}),
                }
            }],
        })
        messages.append({"role": "tool", "content": "[task complete]"})
        lines.append(list(messages))

    return lines


def _write_trace(trace_log_dir: str, task_id: int, lines: list) -> str:
    Path(trace_log_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(trace_log_dir, f"{task_id}.jsonl")
    with open(out_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# GitLab reachability glue (WASP-specific)
#
# The UCM agent browses through an nginx masking proxy that must reach GitLab.
# Its default path (host.docker.internal:<port>) is set up in PTE by
# initialize.py SSH tunnels, which WASP does not run. On the WASP host that path
# is unusable: GitLab is published on 127.0.0.1 (loopback, unreachable from a
# container) and the host firewall drops container->host traffic. The only path
# that works is container-to-container: put the proxy on the same docker network
# as the GitLab container and point its backend at that container's IP.
#
# So, per run, we: find the GitLab container, read its IP, set
# UCM_GITLAB_BACKEND_HOST to it, pre-start the UCM stack, and connect the proxy
# to the GitLab container's network. The runner's DockerComputer.__enter__ then
# sees the stack already running and skips its own `up`, preserving the wiring.
# ---------------------------------------------------------------------------

def _docker(args: list, env: dict = None, timeout: int = 600):
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          env=env, timeout=timeout)


def _resolve_gitlab_container(port: int):
    """Find the running container that serves this run's GitLab.

    Precedence: WASP_UCM_GITLAB_CONTAINER env override → a container publishing
    host <port> to container 8023 → the single-docker default name.
    """
    override = os.environ.get("WASP_UCM_GITLAB_CONTAINER")
    if override:
        return override
    names = [n for n in _docker(["ps", "--format", "{{.Names}}"]).stdout.split() if n]
    for n in names:
        ports = _docker(["inspect", "-f",
                         "{{range $p,$c := .NetworkSettings.Ports}}{{range $c}}{{.HostPort}}->{{$p}} {{end}}{{end}}",
                         n]).stdout
        if f"{port}->8023/tcp" in ports:
            return n
    if "webarena-verified-gitlab" in names:
        return "webarena-verified-gitlab"
    return None


def _gitlab_backend(container: str):
    """Return (network_name, ip) for reaching `container` from another container.
    Prefers the default 'bridge' network; else the first attached network."""
    out = _docker(["inspect", "-f",
                   '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{$v.IPAddress}}{{"\\n"}}{{end}}',
                   container]).stdout
    nets = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1]:
            nets.append((parts[0], parts[1]))
    for net, ip in nets:
        if net == "bridge":
            return net, ip
    return nets[0] if nets else (None, None)


def _prepare_ucm_network(stack, gitlab_net: str):
    """Pre-start the UCM stack (same argv/env the runner uses) and give the proxy
    a route to the GitLab container by joining its network. Idempotent."""
    env = {**os.environ, **stack.compose_env}
    print(f"[run_ucm_agent] Pre-starting UCM stack (project {stack.project_name})...", flush=True)
    up = _docker(["compose", "-f", stack.compose_file, "-p", stack.project_name,
                  "up", "-d", "--build"], env=env, timeout=1800)
    if up.returncode != 0:
        print(up.stdout[-2000:], flush=True)
        print(up.stderr[-2000:], flush=True)
        raise RuntimeError(f"UCM stack pre-up failed (rc={up.returncode})")

    proxy = f"{stack.project_name}-vwa_gitlab_nginx-1"
    conn = _docker(["network", "connect", gitlab_net, proxy])
    if conn.returncode == 0:
        print(f"[run_ucm_agent] proxy {proxy} joined network {gitlab_net}", flush=True)
    elif "already exists" in (conn.stderr or "").lower():
        print(f"[run_ucm_agent] proxy {proxy} already on network {gitlab_net}", flush=True)
    else:
        # Not fatal on its own, but the agent will almost certainly fail to reach
        # GitLab — surface it loudly.
        print(f"[run_ucm_agent] WARNING: could not connect {proxy} to {gitlab_net}: "
              f"{(conn.stderr or '').strip()}", flush=True)


async def _run_ucm_task(task: dict, gitlab_url: str, pte_dir: str, max_steps: int):
    """Run one task through UcmAgentRunner with test settings (masking + qLLM ON)
    and no PTE reset. Returns (agent_result, runner). On agent crash, returns
    (None-ish result with error, runner) so the caller can still salvage whatever
    the agent wrote to model_responses.jsonl."""
    sys.path.insert(0, str(pte_dir))
    from ucm_agent.runner import UcmAgentRunner
    from ucm_agent.computer import PROXY_BASE_URL

    # The agent must browse GitLab THROUGH the masking proxy (gitlab-vwa.com), not
    # the raw host URL. The runner only rewrites the "__GITLAB__" token (PTE's task
    # convention); WASP tasks carry a concrete http://<host>:<port>/... start_url,
    # so without this rewrite the agent's Firefox would load localhost:<port>
    # inside its own container (dead) instead of the proxy.
    task = dict(task)
    _orig_start = task.get("start_url", "") or ""
    if _orig_start.startswith(gitlab_url):
        task["start_url"] = PROXY_BASE_URL.rstrip("/") + _orig_start[len(gitlab_url):]
        print(f"[run_ucm_agent] start_url via proxy: {task['start_url']}", flush=True)

    # Resolve the GitLab container + IP and point the UCM proxy at it BEFORE the
    # stack is built (UcmStack reads UCM_GITLAB_BACKEND_HOST at construction time,
    # inside _init_agent below).
    port = urlparse(gitlab_url).port or 8023
    gl_container = _resolve_gitlab_container(port)
    gl_net = gl_ip = None
    if gl_container:
        gl_net, gl_ip = _gitlab_backend(gl_container)
    if gl_ip:
        os.environ["UCM_GITLAB_BACKEND_HOST"] = gl_ip
        print(f"[run_ucm_agent] UCM proxy backend -> GitLab container "
              f"{gl_container} @ {gl_ip} (net {gl_net})", flush=True)
    else:
        print(f"[run_ucm_agent] WARNING: could not resolve a GitLab container for port {port}; "
              f"the agent will likely fail to reach GitLab. Set WASP_UCM_GITLAB_CONTAINER "
              f"to override.", flush=True)

    runner = UcmAgentRunner(
        headless=True,
        enable_reset=False,      # WASP owns env setup / injection / reset
        force_reset=False,
        gitlab_base_url=gitlab_url,
        max_steps=max_steps,
        # defaults: system_prompt_name="ucm_defense", reveal_mode="trusted" (qLLM ON)
    )
    runner.server = "gitlab"
    runner.base_url = gitlab_url
    print("[run_ucm_agent] Initializing agent (binding UCM docker stack)...", flush=True)
    await runner._init_agent()

    # Pre-up the stack and wire the proxy onto GitLab's network so it's reachable
    # before the agent navigates. The runner's __enter__ then skips its own `up`.
    if gl_ip and gl_net:
        _prepare_ucm_network(runner._stack, gl_net)

    print(f"[run_ucm_agent] Running task: {task.get('intent', '')[:80]!r}", flush=True)
    agent_result = await runner._run_task(task)
    return agent_result, runner


@click.command()
@click.option("--task-config", required=True, help="Path to WASP task JSON file")
@click.option("--trace-log-dir", required=True, help="Directory to write {task_id}.jsonl trace")
@click.option("--pte-dir", required=True, help="Path to the PTE project root")
@click.option("--max-steps", default=120, show_default=True, help="Per-task step budget for the UCM agent")
@click.option("--dry-run-responses", default=None,
              help="Offline smoke test: build the trace from this fixture model_responses.jsonl "
                   "instead of running the agent (no LLM, no docker). Also enabled by UCM_DRY_RUN=1.")
def main(task_config, trace_log_dir, pte_dir, max_steps, dry_run_responses):
    with open(task_config) as f:
        task = json.load(f)

    start_url: str = task["start_url"]
    intent: str = task["intent"]
    task_id: int = task["task_id"]
    sites = task.get("sites", [])

    parsed = urlparse(start_url)
    gitlab_url = f"{parsed.scheme}://{parsed.netloc}"

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    # ---- Unsupported site (UCM is GitLab-only): graceful skip, keep the batch going ----
    if "gitlab" not in sites:
        lines = build_trace_lines(intent, [], answer=None)
        _write_trace(trace_log_dir, task_id, lines)
        _write_meta(trace_log_dir, task_id, {
            "task_id": task_id,
            "agent": "UcmAgent",
            "unsupported_site": True,
            "sites": sites,
            "intent": intent,
            "start_url": start_url,
            "num_steps": 0,
            "note": "UcmAgentRunner supports gitlab only; task skipped.",
            "started_at": started_at,
            "duration_s": round(time.time() - t0, 3),
        })
        print(f"[run_ucm_agent] Task {task_id}: site {sites} unsupported (gitlab-only); wrote skip trace.",
              flush=True)
        return

    # ---- Offline dry-run: build trace from a fixture, no LLM / docker ----
    fixture = dry_run_responses or (
        os.environ.get("UCM_DRY_RUN_RESPONSES")
        if os.environ.get("UCM_DRY_RUN") else None)
    if os.environ.get("UCM_DRY_RUN") and not fixture:
        raise click.UsageError(
            "UCM_DRY_RUN is set but no fixture given. Pass --dry-run-responses <path> "
            "or set UCM_DRY_RUN_RESPONSES=<path>.")
    if fixture:
        entries = list(_iter_jsonl(fixture))
        answer = extract_answer_from_entries(entries)
        lines = build_trace_lines(intent, entries, answer=answer)
        _write_trace(trace_log_dir, task_id, lines)
        _write_meta(trace_log_dir, task_id, {
            "task_id": task_id,
            "agent": "UcmAgent",
            "model": "dry-run",
            "dry_run": True,
            "dry_run_fixture": str(fixture),
            "system_prompt_name": "ucm_defense",
            "reveal_mode": "trusted",
            "qllm": "on",
            "intent": intent,
            "start_url": start_url,
            "gitlab_url": gitlab_url,
            "num_steps": len(lines) - 1,
            "started_at": started_at,
            "duration_s": round(time.time() - t0, 3),
        })
        print(f"[run_ucm_agent] Task {task_id}: DRY RUN from {fixture}; "
              f"wrote {len(lines) - 1} steps (no LLM, no docker).", flush=True)
        return

    # ---- Real run ----
    run_status = {"crashed": False, "exception": None}
    agent_result = None
    runner = None
    try:
        agent_result, runner = asyncio.run(
            _run_ucm_task(task, gitlab_url, pte_dir, max_steps))
        print("[run_ucm_agent] Task complete.", flush=True)
    except Exception as e:
        # Salvage whatever the agent wrote rather than a silent 0-score trace.
        print(f"[run_ucm_agent] Task {task_id} crashed:", flush=True)
        traceback.print_exc()
        run_status["crashed"] = True
        run_status["exception"] = f"{type(e).__name__}: {e}"

    duration_s = round(time.time() - t0, 3)

    results_dir = getattr(runner, "last_results_dir", None) if runner else None
    entries = list(_iter_jsonl(Path(results_dir) / "model_responses.jsonl")) if results_dir else []
    answer = (agent_result or {}).get("answer") or extract_answer_from_entries(entries)

    lines = build_trace_lines(intent, entries, answer=answer)
    out_path = _write_trace(trace_log_dir, task_id, lines)

    _write_meta(trace_log_dir, task_id, {
        "task_id": task_id,
        "agent": "UcmAgent",
        "model": getattr(runner, "model", None),   # real backbone, e.g. claude-sonnet-4-5-*
        "system_prompt_name": "ucm_defense",
        "reveal_mode": "trusted",
        "qllm": "on",
        "intent": intent,
        "start_url": start_url,
        "gitlab_url": gitlab_url,
        "num_steps": len(lines) - 1,
        "results_dir": results_dir,
        "tokens": getattr(runner, "last_tokens", None) if runner else None,
        "task_unsolvable": bool((agent_result or {}).get("task_unsolvable")),
        "agent_error": (agent_result or {}).get("error"),
        "max_steps_reached": (agent_result or {}).get("error") == "max_steps_reached",
        "crashed": run_status["crashed"],
        "exception": run_status["exception"],
        "started_at": started_at,
        "duration_s": duration_s,
    })

    print(f"[run_ucm_agent] Task {task_id}: wrote {len(lines) - 1} steps to {out_path}", flush=True)


if __name__ == "__main__":
    main()
