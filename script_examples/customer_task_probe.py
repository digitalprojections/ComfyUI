"""Summarize a local ComfyUI customer task without changing runtime state."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any


def fetch_json(url: str, timeout: float = 5.0) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return None, str(exc)


def run_nvidia_smi() -> tuple[str, str | None]:
    query = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return "", str(exc)
    return result.stdout.strip(), None


def parse_gpu_csv(output: str) -> dict[str, float] | None:
    first_line = output.splitlines()[0].strip() if output.strip() else ""
    if not first_line:
        return None
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 5:
        return None
    try:
        return {
            "utilization": float(parts[0]),
            "memory_used_mb": float(parts[1]),
            "memory_total_mb": float(parts[2]),
            "power_w": float(parts[3]),
            "temperature_c": float(parts[4]),
        }
    except ValueError:
        return None


def find_running_prompt(queue: dict[str, Any] | None, prompt_id: str | None) -> dict[str, Any] | None:
    for item in (queue or {}).get("queue_running", []):
        if not isinstance(item, list) or len(item) < 2:
            continue
        if prompt_id is None or item[1] == prompt_id:
            return {"number": item[0], "prompt_id": item[1], "workflow": item[2] if len(item) > 2 else {}}
    return None


def history_has_prompt(history: dict[str, Any] | None, prompt_id: str | None) -> bool:
    if not history or not prompt_id:
        return False
    return prompt_id in history


def classify_state(
    *,
    running: bool,
    completed: bool,
    gpu_utilization: float | None,
    agent_processing: bool,
) -> str:
    if completed:
        return "completed"
    if running and gpu_utilization is not None and gpu_utilization >= 20:
        return "active_gpu_work"
    if running and gpu_utilization is not None and gpu_utilization < 5:
        return "likely_stalled_or_waiting"
    if running:
        return "running_unknown_gpu"
    if agent_processing:
        return "agent_processing_but_not_in_comfy_queue"
    return "not_running"


def agent_task(agent_status: dict[str, Any] | None) -> dict[str, Any] | None:
    queue = (agent_status or {}).get("queue", {})
    task = queue.get("currentTask")
    if not isinstance(task, dict):
        return None
    task_id = task.get("id")
    recent_tasks = ((agent_status or {}).get("firebase", {}) or {}).get("recentTasks", [])
    for recent in recent_tasks:
        if isinstance(recent, dict) and recent.get("id") == task_id:
            return {**recent, **task}
    return task


def summarize(
    *,
    queue: dict[str, Any] | None,
    history: dict[str, Any] | None,
    agent_status: dict[str, Any] | None,
    gpu: dict[str, float] | None,
    prompt_id: str | None,
) -> list[str]:
    task = agent_task(agent_status)
    task_id = task.get("id") if task else None
    effective_prompt_id = prompt_id or (task.get("promptId") if task else None)
    running_prompt = find_running_prompt(queue, effective_prompt_id)
    completed = history_has_prompt(history, effective_prompt_id)
    state = classify_state(
        running=running_prompt is not None,
        completed=completed,
        gpu_utilization=gpu.get("utilization") if gpu else None,
        agent_processing=bool(task),
    )

    lines = [
        f"state: {state}",
        f"task: {task_id or 'unknown'}",
        f"prompt: {effective_prompt_id or 'unknown'}",
    ]
    if task:
        lines.append(f"agent: {task.get('status', 'unknown')} {task.get('progress', '?')}/{task.get('progressMax', '?')} - {task.get('statusDetails', '')}")
    queue_pending = len((queue or {}).get("queue_pending", []))
    lines.append(f"comfy: running={'yes' if running_prompt else 'no'}, pending={queue_pending}, history={'yes' if completed else 'no'}")
    if gpu:
        lines.append(
            "gpu: "
            f"{gpu['utilization']:.0f}% util, "
            f"{gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB, "
            f"{gpu['power_w']:.0f} W, {gpu['temperature_c']:.0f} C"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe local ComfyUI and OneImageAgent task state.")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--agent-url", default="http://127.0.0.1:3002")
    parser.add_argument("--prompt-id")
    args = parser.parse_args()

    agent_status, agent_error = fetch_json(f"{args.agent_url.rstrip('/')}/api/status")
    task = agent_task(agent_status)
    prompt_id = args.prompt_id or (task.get("promptId") if task else None)

    queue, queue_error = fetch_json(f"{args.comfy_url.rstrip('/')}/queue")
    history = None
    history_error = None
    if prompt_id:
        history, history_error = fetch_json(f"{args.comfy_url.rstrip('/')}/history/{prompt_id}")
    gpu_text, gpu_error = run_nvidia_smi()
    gpu = parse_gpu_csv(gpu_text)

    for line in summarize(queue=queue, history=history, agent_status=agent_status, gpu=gpu, prompt_id=prompt_id):
        print(line)

    errors = {
        "agent": agent_error,
        "queue": queue_error,
        "history": history_error,
        "gpu": gpu_error,
    }
    for name, error in errors.items():
        if error:
            print(f"{name}_error: {error}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
