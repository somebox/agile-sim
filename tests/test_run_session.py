"""Live `RunSession` stepping with stub client."""

import asyncio
import json
from pathlib import Path

import yaml

from harness.scenario import load_scenario
from harness.web.run_session import SESSIONS, RunSession
from harness.world import build_world_from_scenario, world_from_snapshot

REPO = Path(__file__).resolve().parents[1]


class StubClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._i = 0

    def chat_text(self, model, messages, temperature=0.7, max_tokens=4096):
        if self._i >= len(self._responses):
            raise RuntimeError("no more responses")
        r = self._responses[self._i]
        self._i += 1
        return r, {"input_tokens": 1, "output_tokens": 1, "cost": 0.0}


def _agent(cid: str) -> str:
    return json.dumps(
        {
            "narrative": f"n-{cid}",
            "channel_posts": [{"channel": "#team", "content": f"c-{cid}", "parent_id": None}],
            "vital_self_report": {},
            "work_item_updates": [],
            "process_invocations": [],
        }
    )


def _coach() -> str:
    return json.dumps(
        {"narrative": "c", "channel_posts": [], "vital_nudges": [], "process_invocations": []}
    )


def test_run_session_advance_and_coach_post(tmp_path: Path) -> None:
    async def _go() -> None:
        bundle = load_scenario(REPO / "scenarios" / "two-devs-and-a-pm")
        bundle.scenario["goals"]["max_turns"] = 3

        seq: list[str] = []
        for _ in range(2):
            for cid in ["priya", "marcus", "lin"]:
                seq.append(_agent(cid))
            seq.append(_coach())

        stub = StubClient(seq)
        sess = RunSession.start(
            scenario_dir=bundle.path,
            runs_dir=tmp_path,
            agent_model="stub",
            coach_model="stub",
            coach_mode_cli="human",
            coach_preset_cli=None,
            secrets=None,
            client=stub,
            seed=None,
        )
        key = sess.run_root.name
        SESSIONS[key] = sess

        r1 = await sess.advance()
        assert r1["ok"] is True
        assert r1["stop"] is None

        sess.coach_post(channel=bundle.scenario["channels"][0]["name"], content="human nudge")

        r2 = await sess.advance()
        assert r2["ok"] is True

        msgs = (sess.run_root / "messages.jsonl").read_text(encoding="utf-8")
        assert "human nudge" in msgs
        del SESSIONS[key]

    asyncio.run(_go())


class AutoStubClient:
    """Stub that reports a `served_model` like `openrouter/auto` does."""

    def __init__(self, responses: list[str], served: str) -> None:
        self._responses = list(responses)
        self._served = served
        self._i = 0

    def chat_text(self, model, messages, temperature=0.7, max_tokens=4096):
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r, {"input_tokens": 1, "output_tokens": 1, "cost": 0.0, "served_model": self._served}


def test_run_session_accumulates_served_models(tmp_path: Path) -> None:
    async def _go() -> None:
        bundle = load_scenario(REPO / "scenarios" / "two-devs-and-a-pm")
        bundle.scenario["goals"]["max_turns"] = 2

        seq = [_agent("priya"), _agent("marcus"), _agent("lin"), _coach()]
        stub = AutoStubClient(seq, served="google/gemini-2.0-flash")
        sess = RunSession.start(
            scenario_dir=bundle.path,
            runs_dir=tmp_path,
            agent_model="openrouter/auto",
            coach_model="openrouter/auto",
            coach_mode_cli="human",  # coach makes no LLM call; only agents do
            coach_preset_cli=None,
            secrets=None,
            client=stub,
            seed=None,
        )
        key = sess.run_root.name
        SESSIONS[key] = sess
        try:
            await sess.advance()
            totals = json.loads((sess.run_root / "summary.json").read_text())["totals"]
            assert totals["requested_models"]["agent"] == "openrouter/auto"
            # 3 agents served by the auto-picked model on the first turn.
            assert totals["served_models"]["google/gemini-2.0-flash"] == 3
        finally:
            del SESSIONS[key]

    asyncio.run(_go())


def test_world_from_snapshot_roundtrip(tmp_path: Path) -> None:
    scen = tmp_path / "scenario.yaml"
    scen.write_text(
        yaml.safe_dump(
            {
                "id": "x",
                "channels": [{"name": "#team"}],
                "characters": [
                    {
                        "id": "a",
                        "initial_vitals": {"energy": 5, "motivation": 5, "stress": 5},
                    }
                ],
                "goals": {"max_turns": 2},
            }
        ),
        encoding="utf-8",
    )
    bundle = load_scenario(tmp_path)
    w0 = build_world_from_scenario(bundle.scenario, bundle.character_bodies)
    data = w0.snapshot()
    w1 = world_from_snapshot(data)
    assert w1.turn == w0.turn
    assert w1.characters["a"].vitals == w0.characters["a"].vitals


def test_run_session_resume_cursor_cancel_and_edits(tmp_path: Path) -> None:
    async def _go() -> None:
        bundle = load_scenario(REPO / "scenarios" / "two-devs-and-a-pm")
        bundle.scenario["goals"]["max_turns"] = 4
        order = ["priya", "marcus", "lin"]
        seq: list[str] = []
        for _ in range(4):
            for cid in order:
                seq.append(_agent(cid))
            seq.append(_coach())
        stub = StubClient(seq)
        sess = RunSession.start(
            scenario_dir=bundle.path,
            runs_dir=tmp_path,
            agent_model="stub",
            coach_model="stub",
            coach_mode_cli="llm",
            coach_preset_cli=None,
            secrets=None,
            client=stub,
            seed=None,
        )
        r1 = await sess.advance()
        assert r1["ok"] is True
        assert sess.last_completed_turn >= 1

        restored = RunSession.from_run_dir(run_root=sess.run_root, client=stub)
        assert restored.world.turn == sess.last_completed_turn + 1

        restored.cancel()
        r2 = await restored.advance()
        assert r2["stop"] == "cancelled"

        edit = restored.edit_vital(character_id="priya", vital_name="energy", delta=3)
        assert edit["after"] >= edit["before"]
        pedit = restored.edit_parameter(key="review_threshold", value="0.7")
        assert pedit["key"] == "review_threshold"
        rp = restored.write_reflection("note")
        assert rp.is_file()

    asyncio.run(_go())
