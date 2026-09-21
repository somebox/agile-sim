"""In-process live run sessions (web): step turns with same primitive as CLI (`step_turn`)."""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.runner import (
    _append_jsonl,
    _write_messages_jsonl,
    character_turn_order,
    load_api_key,
    load_preset_for_mode,
    merge_coach_mode,
    merge_model_counts,
    prepare_run_dir,
    resolve_coach_preset_path,
    step_turn,
    write_meta,
)
from harness.scenario import ScenarioBundle, load_scenario
from harness.world import (
    World,
    append_channel_message,
    clamp_int,
    build_world_from_scenario,
    goal_abort,
    goal_met,
    primary_team_channel,
    world_from_snapshot,
)


def _write_interim_summary(
    run_root: Path,
    world: World,
    *,
    coach_mode: str,
    preset_id: str | None,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    elapsed_s: float,
    live: bool,
    requested_models: dict[str, str] | None = None,
    served_models: dict[str, int] | None = None,
) -> dict[str, Any]:
    summary = {
        "run_id": run_root.name,
        "final_turn": world.turn,
        "goal_met": goal_met(world),
        "aborted_stress": goal_abort(world),
        "work_done": sum(1 for wi in world.work_items if wi.state == "done"),
        "final_vitals": {k: v.vitals for k, v in world.characters.items()},
        "org": world.org,
        "coach_mode": coach_mode,
        "coach_preset_id": preset_id,
        "live_session": live,
        "totals": {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost": cost,
            "elapsed_s": round(elapsed_s, 3),
            "requested_models": dict(requested_models or {}),
            "served_models": dict(served_models or {}),
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@dataclass
class RunSession:
    run_id: str
    run_root: Path
    bundle: ScenarioBundle
    world: World
    client: Any
    agent_model: str
    coach_model: str
    coach_mode: str
    coach_preset: dict[str, Any] | None
    preset_id: str | None
    channel: str
    order: list[str]
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    served_models: dict[str, int] = field(default_factory=dict)
    started_monotonic: float = 0.0
    finished: bool = False
    stop_reason: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_completed_turn: int = 0
    events: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    def start(
        cls,
        *,
        scenario_dir: Path,
        runs_dir: Path,
        agent_model: str,
        coach_model: str | None,
        coach_mode_cli: str | None,
        coach_preset_cli: Path | None,
        secrets: Path | None,
        client: Any,
        seed: int | None = None,
    ) -> RunSession:
        bundle = load_scenario(scenario_dir.resolve())
        preset_path = resolve_coach_preset_path(coach_preset_cli, bundle)
        mode = merge_coach_mode(coach_mode_cli, bundle.scenario)
        preset_data, preset_effective = load_preset_for_mode(mode, preset_path)
        preset_id = preset_data.get("id") if preset_data else None

        run_root, rid = prepare_run_dir(runs_dir.resolve())
        coach_model_f = coach_model or agent_model
        channel = primary_team_channel(bundle.scenario)
        world = build_world_from_scenario(bundle.scenario, bundle.character_bodies)
        order = character_turn_order(bundle)

        if mode == "preset" and preset_effective and preset_effective.is_file():
            shutil.copy(preset_effective, run_root / "coach_preset_used.yaml")

        extra: dict[str, Any] = {
            "coach_mode": mode,
            "coach_preset_id": preset_id,
            "coach_preset_path": str(preset_effective) if preset_effective else None,
            "live_session": True,
        }
        write_meta(
            run_root / "meta.yaml",
            bundle=bundle,
            agent_model=agent_model,
            coach_model=coach_model_f,
            seed=seed,
            run_id=rid,
            extra=extra,
        )

        timeline = run_root / "timeline.jsonl"
        _append_jsonl(
            timeline,
            {
                "kind": "run_start",
                "scenario_id": bundle.scenario.get("id"),
                "agent_model": agent_model,
                "coach_model": coach_model_f,
                "coach_mode": mode,
                "coach_preset_id": preset_id,
                "seed": seed,
                "live_session": True,
            },
        )

        import time as _time

        sess = cls(
            run_id=rid,
            run_root=run_root,
            bundle=bundle,
            world=world,
            client=client,
            agent_model=agent_model,
            coach_model=coach_model_f,
            coach_mode=mode,
            coach_preset=preset_data,
            preset_id=preset_id,
            channel=channel,
            order=order,
            started_monotonic=_time.perf_counter(),
        )
        _write_interim_summary(
            run_root,
            world,
            coach_mode=mode,
            preset_id=preset_id,
            tokens_in=0,
            tokens_out=0,
            cost=0.0,
            elapsed_s=0.0,
            live=True,
            requested_models={"agent": agent_model, "coach": coach_model_f},
            served_models={},
        )
        sess.last_completed_turn = 0
        return sess

    @classmethod
    def from_run_dir(
        cls,
        *,
        run_root: Path,
        client: Any,
        secrets: Path | None = None,
    ) -> RunSession:
        """Rebuild session from disk (last snapshot + meta). For page refresh."""

        run_root = run_root.resolve()
        meta_path = run_root / "meta.yaml"
        if not meta_path.is_file():
            raise FileNotFoundError("missing meta.yaml")
        import yaml

        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        sp = meta.get("scenario_path")
        if not sp:
            raise ValueError("meta missing scenario_path")
        bundle = load_scenario(Path(str(sp)))
        agent_model = (meta.get("models") or {}).get("agent") or "stub"
        coach_model = (meta.get("models") or {}).get("coach") or agent_model
        mode = str(meta.get("coach_mode") or merge_coach_mode(None, bundle.scenario)).lower()
        preset_id = meta.get("coach_preset_id")

        snap_path = run_root / "snapshots.jsonl"
        world_data: dict[str, Any] | None = None
        last_snap_turn = 0
        if snap_path.is_file():
            for line in snap_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                w = row.get("world")
                if isinstance(w, dict):
                    world_data = w
                    last_snap_turn = int(row.get("turn") or 0)
        if world_data is None:
            world = build_world_from_scenario(bundle.scenario, bundle.character_bodies)
        else:
            world = world_from_snapshot(world_data)

        summaries_path = run_root / "summary.json"
        tokens_in = tokens_out = 0
        cost = 0.0
        served_models: dict[str, int] = {}
        if summaries_path.is_file():
            try:
                s = json.loads(summaries_path.read_text(encoding="utf-8"))
                t = s.get("totals") or {}
                tokens_in = int(t.get("input_tokens") or 0)
                tokens_out = int(t.get("output_tokens") or 0)
                cost = float(t.get("cost") or 0)
                merge_model_counts(served_models, t.get("served_models") or {})
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass

        if client is None:
            from harness.integrations.openrouter import OpenRouterClient

            client = OpenRouterClient(api_key=load_api_key(secrets))

        channel = primary_team_channel(bundle.scenario)
        order = character_turn_order(bundle)
        import time as _time

        finished = goal_met(world) or goal_abort(world) or world.turn > world.max_turns
        if last_snap_turn > 0 and not finished:
            world.turn = min(world.max_turns + 1, last_snap_turn + 1)

        sess = cls(
            run_id=run_root.name,
            run_root=run_root,
            bundle=bundle,
            world=world,
            client=client,
            agent_model=str(agent_model),
            coach_model=str(coach_model),
            coach_mode=mode,
            coach_preset=None,
            preset_id=preset_id,
            channel=channel,
            order=order,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            served_models=served_models,
            started_monotonic=_time.perf_counter(),
            finished=finished,
            stop_reason="restored" if finished else None,
            last_completed_turn=last_snap_turn,
        )
        return sess

    def _elapsed(self) -> float:
        import time as _time

        return _time.perf_counter() - self.started_monotonic

    def _persist_summary(self) -> dict[str, Any]:
        return _write_interim_summary(
            self.run_root,
            self.world,
            coach_mode=self.coach_mode,
            preset_id=self.preset_id,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cost=self.cost,
            elapsed_s=self._elapsed(),
            live=not self.finished,
            requested_models={"agent": self.agent_model, "coach": self.coach_model},
            served_models=self.served_models,
        )

    def _append_run_end(self, summary: dict[str, Any]) -> None:
        summary["live_session"] = False
        (self.run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _append_jsonl(self.run_root / "timeline.jsonl", {"kind": "run_end", "summary": summary})

    async def advance(self) -> dict[str, Any]:
        async with self.lock:
            if self.finished:
                return {"ok": False, "error": "run finished"}
            if self.world.turn > self.world.max_turns:
                summary = self._persist_summary()
                summary["live_session"] = False
                self._append_run_end(summary)
                self.finished = True
                self.stop_reason = "max_turns"
                return {"ok": False, "error": "max turns reached"}

            loop = asyncio.get_running_loop()
            self.events.put_nowait({"kind": "advance_start", "turn": self.world.turn})

            def _emit(event: dict[str, Any]) -> None:
                loop.call_soon_threadsafe(self.events.put_nowait, event)

            du, stop = await asyncio.to_thread(
                step_turn,
                self.world,
                self.bundle,
                self.client,
                agent_model=self.agent_model,
                coach_model=self.coach_model,
                run_root=self.run_root,
                vital_delta_cap=8,
                coach_nudge_cap=10,
                coach_mode=self.coach_mode,
                coach_preset=self.coach_preset,
                preset_id=self.preset_id,
                channel=self.channel,
                order=self.order,
                verbose=False,
                progress_prefix="",
                on_progress=_emit,
                should_cancel=self.cancel_event.is_set,
            )
            self.tokens_in += int(du["input_tokens"])
            self.tokens_out += int(du["output_tokens"])
            self.cost += float(du["cost"])
            merge_model_counts(self.served_models, du.get("served_models") or {})
            summary = self._persist_summary()

            if stop == "cancelled":
                self.events.put_nowait({"kind": "advance_cancelled", "turn": self.world.turn})
                return {"ok": True, "stop": "cancelled", "turn": self.world.turn}
            if stop in ("goal_met", "aborted_stress"):
                self.last_completed_turn = self.world.turn
            elif self.world.turn > self.world.max_turns:
                self.last_completed_turn = self.world.max_turns
            else:
                self.last_completed_turn = self.world.turn - 1

            if stop:
                self.finished = True
                self.stop_reason = stop
                self._append_run_end(summary)
                self.events.put_nowait({"kind": "advance_stop", "turn": self.world.turn, "stop": stop})
                return {"ok": True, "stop": stop, "turn": self.world.turn}
            if self.world.turn > self.world.max_turns:
                self.finished = True
                self.stop_reason = "max_turns"
                self._append_run_end(summary)
                self.events.put_nowait(
                    {"kind": "advance_stop", "turn": self.world.max_turns, "stop": "max_turns"}
                )
                return {"ok": True, "stop": "max_turns", "turn": self.world.turn}
            self.events.put_nowait({"kind": "advance_done", "turn": self.world.turn - 1})
            return {"ok": True, "stop": None, "turn": self.world.turn}

    def coach_post(self, *, channel: str, content: str, author: str = "coach") -> None:
        ch = channel.strip()
        body = (content or "").strip()
        if not body:
            raise ValueError("empty message")
        append_channel_message(self.world, author, ch, body)
        msgs_path = self.run_root / "messages.jsonl"
        _write_messages_jsonl(msgs_path, self.world.messages)
        _append_jsonl(
            self.run_root / "timeline.jsonl",
            {
                "kind": "coach_human_post",
                "turn": self.world.turn,
                "channel": ch,
                "author": author,
                "content": body,
            },
        )

    def cancel(self) -> None:
        self.cancel_event.set()

    def write_reflection(self, content: str) -> Path:
        body = (content or "").strip()
        p = self.run_root / "reflection.md"
        p.write_text((body + "\n") if body else "", encoding="utf-8")
        _append_jsonl(
            self.run_root / "timeline.jsonl",
            {"kind": "coach_reflection", "turn": self.world.turn, "chars": len(body)},
        )
        return p

    def record_scenario_edit(
        self,
        *,
        target: str,
        payload: dict[str, Any],
        effective_turn: int | None = None,
    ) -> None:
        """Append a structured coach_edit row for scenario-editor changes."""

        eff = effective_turn if effective_turn is not None else self.world.turn + 1
        _append_jsonl(
            self.run_root / "timeline.jsonl",
            {
                "kind": "coach_edit",
                "turn": self.world.turn,
                "target": target,
                "payload": payload,
                "effective_turn": eff,
            },
        )

    def edit_vital(self, *, character_id: str, vital_name: str, delta: int, max_abs: int = 8) -> dict[str, Any]:
        cid = str(character_id).strip()
        vn = str(vital_name).strip()
        if cid not in self.world.characters:
            raise ValueError("unknown character")
        if vn not in self.world.characters[cid].vitals:
            raise ValueError("unknown vital")
        d = max(-max_abs, min(max_abs, int(delta)))
        before = int(self.world.characters[cid].vitals[vn])
        after = clamp_int(before + d)
        self.world.characters[cid].vitals[vn] = after
        _append_jsonl(
            self.run_root / "timeline.jsonl",
            {
                "kind": "coach_edit",
                "turn": self.world.turn,
                "target": "vital",
                "character_id": cid,
                "vital_name": vn,
                "delta": d,
                "before": before,
                "after": after,
                "effective_turn": self.world.turn + 1,
            },
        )
        self._persist_summary()
        return {"character_id": cid, "vital_name": vn, "before": before, "after": after, "delta": d}

    def edit_parameter(self, *, key: str, value: Any) -> dict[str, Any]:
        params = self.bundle.scenario.setdefault("parameters", {})
        if not isinstance(params, dict):
            self.bundle.scenario["parameters"] = {}
            params = self.bundle.scenario["parameters"]
        k = str(key).strip()
        before = params.get(k)
        params[k] = value
        _append_jsonl(
            self.run_root / "timeline.jsonl",
            {
                "kind": "coach_edit",
                "turn": self.world.turn,
                "target": "parameter",
                "key": k,
                "before": before,
                "after": value,
                "effective_turn": self.world.turn + 1,
            },
        )
        return {"key": k, "before": before, "after": value}


SESSIONS: dict[str, RunSession] = {}
