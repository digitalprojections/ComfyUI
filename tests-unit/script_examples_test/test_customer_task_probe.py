from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "script_examples" / "customer_task_probe.py"
SPEC = importlib.util.spec_from_file_location("customer_task_probe", MODULE_PATH)
customer_task_probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(customer_task_probe)


def test_parse_gpu_csv_extracts_first_device_metrics():
    gpu = customer_task_probe.parse_gpu_csv("100, 15900, 16303, 94.42, 50")

    assert gpu == {
        "utilization": 100.0,
        "memory_used_mb": 15900.0,
        "memory_total_mb": 16303.0,
        "power_w": 94.42,
        "temperature_c": 50.0,
    }


def test_classify_active_gpu_work():
    state = customer_task_probe.classify_state(
        running=True,
        completed=False,
        gpu_utilization=100,
        agent_processing=True,
    )

    assert state == "active_gpu_work"


def test_classify_low_gpu_running_as_likely_stalled():
    state = customer_task_probe.classify_state(
        running=True,
        completed=False,
        gpu_utilization=0,
        agent_processing=True,
    )

    assert state == "likely_stalled_or_waiting"


def test_summarize_prefers_agent_prompt_id():
    lines = customer_task_probe.summarize(
        queue={"queue_running": [[1, "prompt-1", {}, {}, []]], "queue_pending": []},
        history={},
        agent_status={
            "queue": {
                "currentTask": {
                    "id": "task-1",
                    "promptId": "prompt-1",
                    "status": "processing",
                    "progress": 30,
                    "progressMax": 100,
                    "statusDetails": "Executing workflow...",
                }
            }
        },
        gpu={
            "utilization": 100,
            "memory_used_mb": 15900,
            "memory_total_mb": 16303,
            "power_w": 94,
            "temperature_c": 49,
        },
        prompt_id=None,
    )

    assert lines[0] == "state: active_gpu_work"
    assert "task-1" in lines[1]
    assert "prompt-1" in lines[2]


def test_agent_task_merges_recent_prompt_id():
    task = customer_task_probe.agent_task(
        {
            "queue": {"currentTask": {"id": "task-1", "status": "processing"}},
            "firebase": {"recentTasks": [{"id": "task-1", "promptId": "prompt-1"}]},
        }
    )

    assert task["promptId"] == "prompt-1"
    assert task["status"] == "processing"
