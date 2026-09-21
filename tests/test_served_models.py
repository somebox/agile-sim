"""Served-model tracking: openrouter/auto reports the concrete model that ran."""

import json
from pathlib import Path

from harness.runner import run_once as run_once_full
from harness.scenario import load_scenario

REPO = Path(__file__).resolve().parents[1]


class AutoStubClient:
    """Mimics OpenRouter: echoes a `served_model` in usage, like `openrouter/auto`."""

    def __init__(self, responses: list[str], served: list[str]) -> None:
        self._responses = list(responses)
        self._served = list(served)
        self._i = 0

    def chat_text(self, model, messages, temperature=0.7, max_tokens=4096):
        r = self._responses[self._i % len(self._responses)]
        served = self._served[self._i % len(self._served)]
        self._i += 1
        return r, {
            "input_tokens": 3,
            "output_tokens": 2,
            "cost": 0.0001,
            "served_model": served,
        }


def _agent_json(cid: str) -> str:
    return json.dumps(
        {
            "narrative": f"{cid} ok",
            "channel_posts": [],
            "vital_self_report": {},
            "work_item_updates": [],
            "process_invocations": [],
        }
    )


def _coach_json() -> str:
    return json.dumps(
        {"narrative": "ok", "channel_posts": [], "vital_nudges": [], "process_invocations": []}
    )


def test_served_models_recorded_in_summary_and_log(tmp_path: Path) -> None:
    bundle = load_scenario(REPO / "scenarios" / "two-devs-and-a-pm")
    bundle.scenario["goals"]["max_turns"] = 1

    # 3 agents + 1 coach per turn; route agents to gemini, coach to claude.
    responses = [_agent_json("a"), _agent_json("b"), _agent_json("c"), _coach_json()]
    served = [
        "google/gemini-2.0-flash",
        "google/gemini-2.0-flash",
        "google/gemini-2.0-flash",
        "anthropic/claude-3.5-sonnet",
    ]
    client = AutoStubClient(responses, served)

    run_root = tmp_path / "run"
    run_root.mkdir()
    summary = run_once_full(
        bundle,
        client,
        agent_model="openrouter/auto",
        coach_model="openrouter/auto",
        run_root=run_root,
        seed=1,
        coach_mode="llm",
        coach_preset=None,
    )

    totals = summary["totals"]
    assert totals["requested_models"] == {"agent": "openrouter/auto", "coach": "openrouter/auto"}
    assert totals["served_models"]["google/gemini-2.0-flash"] == 3
    assert totals["served_models"]["anthropic/claude-3.5-sonnet"] == 1

    # llm_calls.jsonl carries the served_model per call.
    rows = [
        json.loads(line)
        for line in (run_root / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    agent_rows = [r for r in rows if r["role"] == "agent"]
    assert agent_rows and all(r["served_model"] == "google/gemini-2.0-flash" for r in agent_rows)
    coach_rows = [r for r in rows if r["role"] == "coach"]
    assert coach_rows and coach_rows[0]["served_model"] == "anthropic/claude-3.5-sonnet"
