"""Orchestrate one harness run and persist JSONL artifacts."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from harness import __version__ as harness_version
from harness import engine
from harness.agent import run_agent_turn
from harness.coach import run_coach_turn
from harness.coach_preset import coach_turn_from_preset, load_coach_preset
from harness.scenario import ScenarioBundle, load_scenario
from harness.schemas import AgentTurnOutput, CoachTurnOutput
from harness.world import (
    World,
    append_channel_message,
    apply_coach_nudges,
    apply_vital_deltas,
    apply_work_updates,
    build_world_from_scenario,
    can_post,
    goal_abort,
    goal_met,
    primary_team_channel,
)


def merge_model_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    """Accumulate served-model call counts from ``src`` into ``dst`` in place."""

    for model, n in (src or {}).items():
        if model:
            dst[str(model)] = dst.get(str(model), 0) + int(n)


def record_served_model(counts: dict[str, int], usage: dict[str, Any]) -> str | None:
    """Bump the served-model counter for one call; return the served model id."""

    served = usage.get("served_model")
    if served:
        counts[str(served)] = counts.get(str(served), 0) + 1
    return served


def apply_agent_output(
    world: World,
    char_id: str,
    out: AgentTurnOutput,
    *,
    vital_delta_cap: int,
    channel: str,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    apply_vital_deltas(world, char_id, out.vital_self_report or {}, clamp_delta=vital_delta_cap)
    apply_work_updates(world, char_id, out.work_item_updates or [])
    dm = f"dm/{char_id}"
    for post in out.channel_posts:
        ch = (post.channel or channel).strip()
        body = (post.content or "").strip()
        if not body:
            continue
        ok, reason = can_post(author=char_id, channel=ch, scenario=scenario)
        if not ok:
            rejected.append({"who": char_id, "channel": ch, "reason": reason or "rejected"})
            continue
        if ch == channel:
            append_channel_message(world, char_id, channel, body)
        elif ch == dm:
            append_channel_message(world, char_id, dm, body)
        else:
            rejected.append({"who": char_id, "channel": ch, "reason": "not_primary_or_owned_dm"})
    return rejected


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def try_git_sha(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _write_messages_jsonl(path: Path, messages: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(asdict(m), default=str) + "\n")


def load_api_key(secrets_path: Path | None) -> str:
    root = _repo_root()
    path = secrets_path or (root / "secrets.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Missing secrets file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    key = data.get("openrouter_api_key") or data.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(f"{path} must set openrouter_api_key")
    return str(key)


def merge_coach_mode(cli_value: str | None, scenario: dict[str, Any]) -> str:
    """CLI overrides scenario `harness.coach_mode` when provided."""

    if cli_value is not None and str(cli_value).strip():
        return str(cli_value).strip().lower()
    h = scenario.get("harness") or {}
    return str(h.get("coach_mode") or "llm").strip().lower()


def resolve_coach_preset_path(cli_path: Path | None, bundle: ScenarioBundle) -> Path | None:
    if cli_path is not None:
        return cli_path.resolve()
    h = bundle.scenario.get("harness") or {}
    rel = h.get("coach_preset_file")
    if rel:
        p = (bundle.path / str(rel)).resolve()
        return p if p.exists() else None
    return None


def load_preset_for_mode(mode: str, preset_path: Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    if mode != "preset":
        return None, preset_path
    if not preset_path or not preset_path.is_file():
        raise ValueError(
            "coach_mode=preset requires a valid preset file "
            "(--coach-preset or scenario harness.coach_preset_file)"
        )
    return load_coach_preset(preset_path), preset_path


def character_turn_order(bundle: ScenarioBundle) -> list[str]:
    return [str(c["id"]) for c in (bundle.scenario.get("characters") or [])]


def _progress(verbose: bool, prefix: str, message: str) -> None:
    if not verbose:
        return
    line = f"{prefix}{message}" if prefix else message
    print(line, file=sys.stderr, flush=True)


def apply_coach_output(
    world: World,
    out: CoachTurnOutput,
    channel: str,
    nudge_cap: int,
    *,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    for post in out.channel_posts:
        ch = (post.channel or channel).strip()
        body = (post.content or "").strip()
        if not body:
            continue
        ok, reason = can_post(author="coach", channel=ch, scenario=scenario)
        if not ok:
            rejected.append({"who": "coach", "channel": ch, "reason": reason or "rejected"})
            continue
        if ch == channel:
            append_channel_message(world, "coach", ch, body)
        elif ch.startswith("dm/"):
            recip = ch[3:].strip()
            if recip in world.characters:
                append_channel_message(world, "coach", ch, body)
            else:
                rejected.append({"who": "coach", "channel": ch, "reason": "dm_unknown_character"})
        else:
            rejected.append({"who": "coach", "channel": ch, "reason": "unknown_target"})
    apply_coach_nudges(world, out.vital_nudges or [], max_abs=nudge_cap)
    return rejected


def step_turn(
    world: World,
    bundle: ScenarioBundle,
    client: Any,
    *,
    agent_model: str,
    coach_model: str,
    run_root: Path,
    vital_delta_cap: int,
    coach_nudge_cap: int,
    coach_mode: str,
    coach_preset: dict[str, Any] | None,
    preset_id: str | None,
    channel: str,
    order: list[str],
    verbose: bool,
    progress_prefix: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[dict[str, int | float], str | None]:
    """Run one simulation turn (agents + coach), persist artifacts, maybe advance `world.turn`.

    Returns ``(usage_delta, stop_reason)`` where ``stop_reason`` is ``goal_met``,
    ``aborted_stress``, or ``None`` (continue / exhausted max turn slot handled by caller
    via ``while world.turn <= world.max_turns``).
    """

    timeline = run_root / "timeline.jsonl"
    snapshots = run_root / "snapshots.jsonl"
    msgs_path = run_root / "messages.jsonl"
    llm_log = run_root / "llm_calls.jsonl"

    tokens_in = 0
    tokens_out = 0
    cost = 0.0
    served_models: dict[str, int] = {}
    cancel_check = should_cancel or (lambda: False)

    def _du() -> dict[str, Any]:
        return {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost": cost,
            "served_models": dict(served_models),
        }

    def emit(event: dict[str, Any]) -> None:
        if on_progress:
            on_progress(event)

    _append_jsonl(timeline, {"kind": "turn_start", "turn": world.turn})
    emit({"kind": "turn_start", "turn": world.turn})
    for ev in engine.tick(world, bundle):
        _append_jsonl(timeline, ev)
    _progress(verbose, progress_prefix, f"--- turn {world.turn} / {world.max_turns} ---")

    for cid in order:
        if cancel_check():
            _append_jsonl(timeline, {"kind": "turn_cancelled", "turn": world.turn, "at": f"agent:{cid}"})
            emit({"kind": "turn_cancelled", "turn": world.turn, "at": f"agent:{cid}"})
            return _du(), "cancelled"
        emit({"kind": "agent_start", "turn": world.turn, "character": cid})
        t0 = time.perf_counter()
        try:
            out, _raw, usage = run_agent_turn(
                client=client,
                model=agent_model,
                bundle=bundle,
                world=world,
                char_id=cid,
                vital_delta_cap=vital_delta_cap,
            )
        except Exception as err:  # noqa: BLE001
            _append_jsonl(
                timeline,
                {"kind": "parse_error", "turn": world.turn, "who": cid, "error": str(err)},
            )
            out = AgentTurnOutput(narrative="(agent call failed)", channel_posts=[])
            usage = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}

        dt_ms = int((time.perf_counter() - t0) * 1000)
        tokens_in += int(usage.get("input_tokens", 0))
        tokens_out += int(usage.get("output_tokens", 0))
        cost += float(usage.get("cost", 0) or 0)
        served = record_served_model(served_models, usage)
        _append_jsonl(
            llm_log,
            {
                "turn": world.turn,
                "role": "agent",
                "character": cid,
                "model": agent_model,
                "served_model": served,
                "usage": usage,
                "latency_ms": dt_ms,
            },
        )
        for rej in apply_agent_output(
            world,
            cid,
            out,
            vital_delta_cap=vital_delta_cap,
            channel=channel,
            scenario=bundle.scenario,
        ):
            _append_jsonl(timeline, {"kind": "post_rejected", "turn": world.turn, **rej})
        for ev in engine.apply_process_invocations(
            out.process_invocations,
            world=world,
            bundle=bundle,
            source=f"agent:{cid}",
            turn=world.turn,
        ):
            _append_jsonl(timeline, ev)
        _append_jsonl(
            timeline,
            {
                "kind": "agent_turn",
                "turn": world.turn,
                "character": cid,
                "narrative": out.narrative,
                "posts": [p.model_dump() for p in out.channel_posts],
                "vital_deltas": out.vital_self_report,
                "work_updates": out.work_item_updates,
            },
        )
        tin, tout = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        cst = float(usage.get("cost", 0) or 0)
        _progress(
            verbose,
            progress_prefix,
            f"  agent  {cid:12}  {dt_ms:5} ms  model={agent_model}  tok {tin}+{tout}  ${cst:.4f}",
        )
        emit({"kind": "agent_done", "turn": world.turn, "character": cid, "latency_ms": dt_ms})

    t0_coach = time.perf_counter()
    if cancel_check():
        _append_jsonl(timeline, {"kind": "turn_cancelled", "turn": world.turn, "at": "coach"})
        emit({"kind": "turn_cancelled", "turn": world.turn, "at": "coach"})
        return _du(), "cancelled"
    emit({"kind": "coach_start", "turn": world.turn, "source": coach_mode})
    coach_logged_model = coach_model
    try:
        if coach_mode == "none":
            cout = CoachTurnOutput()
            usage_c = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            coach_logged_model = "none"
            _append_jsonl(
                llm_log,
                {
                    "turn": world.turn,
                    "role": "coach",
                    "model": coach_logged_model,
                    "usage": usage_c,
                    "latency_ms": 0,
                    "source": "none",
                },
            )
        elif coach_mode == "preset" and coach_preset:
            cout = coach_turn_from_preset(coach_preset, world.turn, default_channel=channel)
            usage_c = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            coach_logged_model = f"preset:{preset_id or 'file'}"
            _append_jsonl(
                llm_log,
                {
                    "turn": world.turn,
                    "role": "coach",
                    "model": coach_logged_model,
                    "usage": usage_c,
                    "latency_ms": int((time.perf_counter() - t0_coach) * 1000),
                    "source": "preset",
                },
            )
        elif coach_mode == "human":
            cout = CoachTurnOutput()
            usage_c = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            coach_logged_model = "human"
            _append_jsonl(
                llm_log,
                {
                    "turn": world.turn,
                    "role": "coach",
                    "model": coach_logged_model,
                    "usage": usage_c,
                    "latency_ms": 0,
                    "source": "human",
                },
            )
        else:
            cout, _raw_c, usage_c = run_coach_turn(
                client=client,
                model=coach_model,
                bundle=bundle,
                world=world,
                nudge_cap=coach_nudge_cap,
            )
            served_c = record_served_model(served_models, usage_c)
            _append_jsonl(
                llm_log,
                {
                    "turn": world.turn,
                    "role": "coach",
                    "model": coach_model,
                    "served_model": served_c,
                    "usage": usage_c,
                    "latency_ms": int((time.perf_counter() - t0_coach) * 1000),
                    "source": "llm",
                },
            )
    except Exception as err:  # noqa: BLE001
        _append_jsonl(
            timeline,
            {"kind": "parse_error", "turn": world.turn, "who": "coach", "error": str(err)},
        )
        cout = CoachTurnOutput()
        usage_c = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}

    coach_dt_ms = int((time.perf_counter() - t0_coach) * 1000)
    tokens_in += int(usage_c.get("input_tokens", 0))
    tokens_out += int(usage_c.get("output_tokens", 0))
    cost += float(usage_c.get("cost", 0) or 0)
    for rej in apply_coach_output(
        world, cout, channel, coach_nudge_cap, scenario=bundle.scenario
    ):
        _append_jsonl(timeline, {"kind": "post_rejected", "turn": world.turn, **rej})
    for ev in engine.apply_process_invocations(
        cout.process_invocations,
        world=world,
        bundle=bundle,
        source="coach",
        turn=world.turn,
    ):
        _append_jsonl(timeline, ev)
    cin_c, cout_c = int(usage_c.get("input_tokens", 0)), int(usage_c.get("output_tokens", 0))
    c_cost = float(usage_c.get("cost", 0) or 0)
    n_cposts = len(cout.channel_posts or [])
    n_cnudges = len(cout.vital_nudges or [])
    _progress(
        verbose,
        progress_prefix,
        f"  coach {str(coach_logged_model):14} {coach_dt_ms:5} ms  "
        f"source={coach_mode}  tok {cin_c}+{cout_c}  ${c_cost:.4f}  "
        f"posts={n_cposts} nudges={n_cnudges}",
    )
    emit(
        {
            "kind": "coach_done",
            "turn": world.turn,
            "source": coach_mode,
            "latency_ms": coach_dt_ms,
        }
    )
    _append_jsonl(
        timeline,
        {
            "kind": "coach_turn",
            "turn": world.turn,
            "source": coach_mode,
            "narrative": cout.narrative,
            "nudges": cout.vital_nudges,
            "posts": [p.model_dump() for p in cout.channel_posts],
        },
    )

    _append_jsonl(snapshots, {"turn": world.turn, "world": world.snapshot()})
    _write_messages_jsonl(msgs_path, world.messages)

    if goal_met(world):
        _progress(verbose, progress_prefix, "stop: goal_met")
        emit({"kind": "turn_stop", "turn": world.turn, "stop": "goal_met"})
        return _du(), "goal_met"
    if goal_abort(world):
        _progress(verbose, progress_prefix, "stop: aborted_stress")
        emit({"kind": "turn_stop", "turn": world.turn, "stop": "aborted_stress"})
        return _du(), "aborted_stress"

    world.turn += 1
    emit({"kind": "turn_done", "turn": world.turn - 1})
    return _du(), None


def run_once(
    bundle: ScenarioBundle,
    client: Any,
    *,
    agent_model: str,
    coach_model: str,
    run_root: Path,
    seed: int | None = None,
    vital_delta_cap: int = 8,
    coach_nudge_cap: int = 10,
    coach_mode: str = "llm",
    coach_preset: dict[str, Any] | None = None,
    coach_preset_source: Path | None = None,
    verbose: bool = False,
    progress_prefix: str = "",
) -> dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    coach_mode = (coach_mode or "llm").strip().lower()
    preset_id = coach_preset.get("id") if coach_preset else None

    channel = primary_team_channel(bundle.scenario)
    world = build_world_from_scenario(bundle.scenario, bundle.character_bodies)
    run_root.mkdir(parents=True, exist_ok=True)

    if coach_mode == "preset" and coach_preset_source and coach_preset_source.is_file():
        shutil.copy(coach_preset_source, run_root / "coach_preset_used.yaml")

    if coach_mode == "preset" and coach_preset is None:
        raise ValueError("coach_mode=preset requires a loaded coach preset")

    timeline = run_root / "timeline.jsonl"

    tokens_in = 0
    tokens_out = 0
    cost = 0.0
    served_models: dict[str, int] = {}
    order = character_turn_order(bundle)
    coach_steps = 0 if coach_mode == "none" else 1
    _progress(
        verbose,
        progress_prefix,
        f"harness: scenario={bundle.scenario.get('id')}  "
        f"turns≤{world.max_turns}  {len(order)} agents + {coach_steps} coach step(s)/turn  "
        f"coach_mode={coach_mode}  channel={channel}",
    )

    _append_jsonl(
        timeline,
        {
            "kind": "run_start",
            "scenario_id": bundle.scenario.get("id"),
            "agent_model": agent_model,
            "coach_model": coach_model,
            "coach_mode": coach_mode,
            "coach_preset_id": preset_id,
            "seed": seed,
        },
    )

    start_clock = time.perf_counter()

    while world.turn <= world.max_turns:
        du, stop = step_turn(
            world,
            bundle,
            client,
            agent_model=agent_model,
            coach_model=coach_model,
            run_root=run_root,
            vital_delta_cap=vital_delta_cap,
            coach_nudge_cap=coach_nudge_cap,
            coach_mode=coach_mode,
            coach_preset=coach_preset,
            preset_id=preset_id,
            channel=channel,
            order=order,
            verbose=verbose,
            progress_prefix=progress_prefix,
        )
        tokens_in += int(du["input_tokens"])
        tokens_out += int(du["output_tokens"])
        cost += float(du["cost"])
        merge_model_counts(served_models, du.get("served_models") or {})
        if stop:
            break

    if verbose and not goal_met(world) and not goal_abort(world):
        _progress(
            verbose,
            progress_prefix,
            f"stop: max_turns (last completed turn {world.max_turns})",
        )

    elapsed_s = time.perf_counter() - start_clock

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
        "totals": {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost": cost,
            "elapsed_s": round(elapsed_s, 3),
            "requested_models": {"agent": agent_model, "coach": coach_model},
            "served_models": served_models,
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _append_jsonl(timeline, {"kind": "run_end", "summary": summary})
    _progress(
        verbose,
        progress_prefix,
        "harness: finished  "
        f"elapsed={elapsed_s:.1f}s  tok={tokens_in}+{tokens_out}  "
        f"${cost:.4f}  goal_met={summary['goal_met']}",
    )
    return summary


def prepare_run_dir(out_base: Path, name_prefix: str = "run") -> tuple[Path, str]:
    rid = f"{name_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    return out_base / rid, rid


def prepare_named_run_dir(parent: Path, name: str) -> tuple[Path, str]:
    """Create a deterministic run directory under parent (e.g. batch run index)."""

    run_root = parent / name
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root, name


def write_meta(
    path: Path,
    *,
    bundle: ScenarioBundle,
    agent_model: str,
    coach_model: str,
    seed: int | None,
    run_id: str,
    extra: dict[str, Any] | None = None,
) -> None:
    root = _repo_root()
    meta: dict[str, Any] = {
        "run_id": run_id,
        "harness_version": harness_version,
        "git_sha": try_git_sha(root),
        "scenario_id": bundle.scenario.get("id"),
        "scenario_path": str(bundle.path),
        "models": {"agent": agent_model, "coach": coach_model},
        "seed": seed,
    }
    if extra:
        meta.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def dispatch_run(
    scenario_dir: Path,
    *,
    out: Path,
    agent_model: str,
    coach_model: str | None = None,
    seed: int | None = None,
    secrets: Path | None = None,
    run_name: str = "run",
    meta_extra: dict[str, Any] | None = None,
    coach_mode_cli: str | None = None,
    coach_preset_cli: Path | None = None,
    verbose: bool = False,
    progress_prefix: str | None = None,
) -> dict[str, Any]:
    coach_model = coach_model or agent_model
    bundle = load_scenario(scenario_dir)
    mode = merge_coach_mode(coach_mode_cli, bundle.scenario)
    preset_path = resolve_coach_preset_path(coach_preset_cli, bundle)
    preset_data, preset_effective = load_preset_for_mode(mode, preset_path)

    run_root, rid = prepare_run_dir(out, name_prefix=run_name)
    pp = progress_prefix if progress_prefix is not None else (f"[{rid}] " if verbose else "")
    extra: dict[str, Any] = dict(meta_extra or {})
    extra.update(
        {
            "coach_mode": mode,
            "coach_preset_id": preset_data.get("id") if preset_data else None,
            "coach_preset_path": str(preset_effective) if preset_effective else None,
        }
    )
    write_meta(
        run_root / "meta.yaml",
        bundle=bundle,
        agent_model=agent_model,
        coach_model=coach_model,
        seed=seed,
        run_id=rid,
        extra=extra,
    )

    from harness.integrations.openrouter import OpenRouterClient

    key = load_api_key(secrets)
    client = OpenRouterClient(api_key=key)

    return run_once(
        bundle,
        client,
        agent_model=agent_model,
        coach_model=coach_model,
        run_root=run_root,
        seed=seed,
        coach_mode=mode,
        coach_preset=preset_data,
        coach_preset_source=preset_effective,
        verbose=verbose,
        progress_prefix=pp,
    )
