"""FastAPI app: read-only mission control + experiments."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from harness.analyse import analyse_batch, batch_metrics, load_meta
from harness.runner import load_api_key
from harness.scenario import list_scenarios
from harness.web.render import md_to_html
from harness.web.resolve import (
    ResolvedBatch,
    ResolvedRun,
    resolve_slug_under_runs,
    run_url_path,
)
from harness.web.run_reader import (
    agent_narrative_snippets,
    build_picker_entries,
    channel_meta_for,
    list_batches,
    list_runs,
    load_run,
    recent_posts_for_character,
    scenario_from_bundle_meta,
    svg_sparkline,
    vitals_history_for_character,
    work_item_timeline_events,
)
from harness.web.run_session import SESSIONS, RunSession
from harness.web.scenario_io import (
    character_md_path,
    create_character,
    delete_character,
    delete_channel,
    delete_team,
    delete_work_item,
    is_shared_character_path,
    load_scenario_yaml,
    lock_state,
    process_yaml_path,
    read_setting_md,
    read_yaml_text,
    upsert_channel,
    upsert_team,
    upsert_work_item,
    update_goals,
    update_parameters,
    upsert_character,
    validate_character_id,
    write_yaml_file_safely,
    best_practices_yaml_path,
    write_character_backstory,
    write_setting_md,
)
from harness.world import parse_mentions


def _runs_dir_kw(runs_dir: Path | None) -> Path:
    if runs_dir is not None:
        return runs_dir.resolve()
    return Path(os.environ.get("AGILE_SIM_RUNS_DIR", "runs")).resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_scenario_dir(scenarios_root: Path, slug: str) -> Path:
    base = scenarios_root.resolve()
    cand = (base / slug.strip()).resolve()
    try:
        cand.relative_to(base)
    except ValueError as err:
        raise HTTPException(status_code=400, detail="invalid scenario path") from err
    if not cand.is_dir() or not (cand / "scenario.yaml").is_file():
        raise HTTPException(status_code=404, detail="scenario not found")
    return cand


def _scenario_cards(scenarios_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in list_scenarios(scenarios_root):
        rows.append(
            {
                "slug": s.slug,
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "characters_count": s.characters_count,
                "channels_count": s.channels_count,
                "cover_url": s.cover_url,
            }
        )
    return rows


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip("-_")
    return text


_EDITOR_SECTIONS = (
    "setting",
    "characters",
    "teams",
    "channels",
    "work_items",
    "goals",
    "parameters",
    "process",
    "best_practices",
)


def _yaml_line_rows(text: str, error_line: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = (text or "").splitlines() or [""]
    marker = error_line
    if marker is not None and marker > len(lines):
        marker = len(lines)
    for idx, line in enumerate(lines, start=1):
        rows.append({"no": idx, "text": line, "error": bool(marker and idx == marker)})
    return rows


def _snapshot_at(bundle: Any, turn: int | None) -> dict[str, Any] | None:
    if turn is None:
        if bundle.snapshots:
            return bundle.snapshots[-1].world
        return {}
    for s in reversed(bundle.snapshots):
        if s.turn <= turn:
            return s.world
    return bundle.snapshots[0].world if bundle.snapshots else {}


def _goal_stoplight(summary: dict[str, Any]) -> str:
    if summary.get("aborted_stress"):
        return "fail"
    if summary.get("goal_met"):
        return "ok"
    return "warn"


def _runner_ctx(
    bundle: Any,
    snap_world: dict[str, Any] | None,
    goal_status: str,
    eff_turn: int,
    max_turn: int,
) -> dict[str, Any]:
    from harness.web.sprites import (
        character_meta_by_id,
        character_sprite_set,
        expression_from_vitals,
        sprite_url,
    )

    scen = scenario_from_bundle_meta(bundle.meta)
    char_meta = character_meta_by_id(scen)
    snap = snap_world or {}
    wis = snap.get("work_items") or []
    by_id = {str(wi.get("id")): wi for wi in wis}
    goals_cfg = scen.get("goals") or {}
    req_ids = list(goals_cfg.get("require_done_ids") or [])
    goal_rows: list[dict[str, Any]] = []
    for rid in req_ids:
        rid_s = str(rid)
        wi = by_id.get(rid_s, {})
        st = str(wi.get("state") or "—").lower()
        sl = "ok" if st == "done" else "warn"
        goal_rows.append(
            {
                "id": rid_s,
                "title": str(wi.get("title") or rid_s),
                "state": st,
                "stoplight": sl,
            }
        )
    live_turn = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
    team_members: dict[str, list[str]] = {}
    for tm in scen.get("teams") or []:
        tid = str(tm.get("id", ""))
        if tid:
            team_members[tid] = [str(x) for x in (tm.get("member_ids") or [])]
    return {
        "char_meta": char_meta,
        "scenario": scen,
        "goal_rows": goal_rows,
        "require_done_ids": req_ids,
        "work_done_count": sum(1 for w in wis if str(w.get("state")).lower() == "done"),
        "work_total": len(wis),
        "live_turn": live_turn,
        "viewing_replay": eff_turn != live_turn,
        "max_turn_ui": max_turn,
        "sprite_set": "default",
        "character_sprite_set": character_sprite_set,
        "expression_from_vitals": expression_from_vitals,
        "sprite_url": sprite_url,
        "team_members": team_members,
        "primary_team_channel": bundle.primary_channel,
    }


def _vitals_spark_map(bundle: Any, chars: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for cid in chars:
        c = str(cid)
        hist_e = [v for _, v in vitals_history_for_character(bundle, c, "energy")]
        hist_m = [v for _, v in vitals_history_for_character(bundle, c, "motivation")]
        hist_s = [v for _, v in vitals_history_for_character(bundle, c, "stress")]
        out[c] = {
            "energy": svg_sparkline(hist_e) if hist_e else "",
            "motivation": svg_sparkline(hist_m) if hist_m else "",
            "stress": svg_sparkline(hist_s) if hist_s else "",
        }
    return out


def _channel_attention_map(bundle: Any, *, since_turn: int) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for c in bundle.channels:
        cid = c.id
        msgs = bundle.messages_by_channel.get(cid, [])
        if not msgs:
            out[cid] = False
            continue
        last = msgs[-1]
        out[cid] = bool(last.turn >= since_turn and str(last.author).lower() == "coach")
    return out


def _channel_message_counts(bundle: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in bundle.channels:
        cid = c.id
        counts[cid] = len(bundle.messages_by_channel.get(cid, []))
    return counts


def _channel_unread_counts(bundle: Any, *, since_turn: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in bundle.channels:
        cid = c.id
        msgs = bundle.messages_by_channel.get(cid, [])
        counts[cid] = sum(1 for m in msgs if int(getattr(m, "turn", 0) or 0) >= since_turn)
    return counts


def _normalize_runner_view(raw: str | None) -> str:
    valid = {"channels", "kanban", "roster", "timeline", "summary"}
    value = str(raw or "channels").strip().lower()
    return value if value in valid else "channels"


def _sentiment_series(bundle: Any) -> tuple[list[int], str]:
    values: list[int] = []
    for snap in bundle.snapshots:
        chars = ((snap.world or {}).get("characters") or {}).values()
        if not chars:
            continue
        pts: list[int] = []
        for ch in chars:
            vitals = (ch or {}).get("vitals") or {}
            try:
                energy = int(vitals.get("energy", 50))
                motivation = int(vitals.get("motivation", 50))
                stress = int(vitals.get("stress", 50))
            except (TypeError, ValueError):
                continue
            composite = int(round((energy + motivation + (100 - stress)) / 3))
            pts.append(max(0, min(100, composite)))
        if pts:
            values.append(int(round(sum(pts) / len(pts))))
    return values, (svg_sparkline(values, width=180, height=38) if values else "")


def _group_timeline_rows(bundle: Any) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in bundle.timeline:
        raw = row.raw or {}
        try:
            turn = int(raw.get("turn", 0) or 0)
        except (TypeError, ValueError):
            turn = 0
        grouped.setdefault(turn, []).append({"kind": row.kind, "raw": raw})
    out: list[dict[str, Any]] = []
    for turn in sorted(grouped.keys()):
        out.append({"turn": turn, "events": grouped[turn]})
    return out


def _roster_delta_map(bundle: Any, chars: dict[str, Any], eff_turn: int) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if not chars:
        return out
    by_turn: dict[int, dict[str, Any]] = {}
    for snap in bundle.snapshots:
        by_turn[int(snap.turn)] = (snap.world or {}).get("characters") or {}
    now_chars = by_turn.get(int(eff_turn), {})
    prev_chars = by_turn.get(max(1, int(eff_turn) - 1), {})
    for cid in chars.keys():
        now_v = (now_chars.get(cid) or {}).get("vitals") or {}
        prev_v = (prev_chars.get(cid) or {}).get("vitals") or {}
        row: dict[str, int] = {}
        for key in ("energy", "motivation", "stress"):
            try:
                row[key] = int(now_v.get(key, 0)) - int(prev_v.get(key, 0))
            except (TypeError, ValueError):
                row[key] = 0
        out[cid] = row
    return out


def create_app(*, runs_dir: Path | None = None, scenarios_dir: Path | None = None) -> FastAPI:
    runs_dir = _runs_dir_kw(runs_dir)
    scenarios_root = scenarios_dir.resolve() if scenarios_dir is not None else (_repo_root() / "scenarios")
    pkg = Path(__file__).resolve().parent
    templates_dir = pkg / "templates"
    static_dir = pkg / "static"

    jinja = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    def _tojson(v: Any, indent: int | None = 2, **_: Any) -> str:
        return json.dumps(v, indent=indent, default=str, ensure_ascii=False)

    jinja.filters["tojson"] = _tojson

    def _render_mentions(text: str, scenario: dict[str, Any] | None, run_url_path: str, turn: int) -> str:
        """Linkify `@cid` mentions in message bodies to open the target's DM."""

        from markupsafe import Markup, escape
        from urllib.parse import quote

        scen = scenario or {}
        ids = {str(c.get("id", "")).strip() for c in (scen.get("characters") or []) if c.get("id")}
        ids = {x for x in ids if x}
        if not text:
            return ""
        if not ids:
            return str(escape(text))
        valid = parse_mentions(text, ids)
        if not valid:
            return str(escape(text))
        valid_lower = {x.lower() for x in valid}
        out_parts: list[str] = []
        last = 0
        for m in re.finditer(r"@([A-Za-z][A-Za-z0-9_-]{0,31})", text):
            cid_raw = m.group(1)
            cid_lc = cid_raw.lower()
            if cid_lc not in valid_lower:
                continue
            out_parts.append(str(escape(text[last : m.start()])))
            dm = f"dm/{cid_lc}"
            url = f"/partials/run/{run_url_path}/channel?channel={quote(dm)}&turn={turn}"
            out_parts.append(
                '<a class="msg-mention" href="#"'
                f' hx-get="{url}" hx-target="#center-stage" hx-swap="innerHTML"'
                f' hx-push-url="/runs/{run_url_path}?view=channels&channel={quote(dm)}"'
                f' title="Open DM with @{escape(cid_raw)}">@{escape(cid_raw)}</a>'
            )
            last = m.end()
        out_parts.append(str(escape(text[last:])))
        return Markup("".join(out_parts))

    jinja.globals["render_mentions"] = _render_mentions

    app = FastAPI(title="agile-sim harness", version="0.1")

    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def render(name: str, request: Request, **ctx: Any) -> HTMLResponse:
        tpl = jinja.get_template(name)
        html = tpl.render(request=request, **ctx)
        return HTMLResponse(html)

    def _scenario_editor_ctx(scen_dir: Path, *, slug: str, section: str) -> dict[str, Any]:
        data = json.loads(json.dumps(load_scenario_yaml(scen_dir)))
        lstate = lock_state(scen_dir, SESSIONS)
        setting_raw = read_setting_md(scen_dir)
        setting_html = md_to_html(setting_raw) if setting_raw else ""
        chars = list(data.get("characters") or [])
        char_by_id = {
            str(c.get("id", "")).strip(): c for c in chars if str(c.get("id", "")).strip()
        }
        process_raw = read_yaml_text(process_yaml_path(scen_dir))
        bp_raw = read_yaml_text(best_practices_yaml_path(scen_dir))
        return {
            "slug": slug,
            "scenario": data,
            "section": section,
            "sections": _EDITOR_SECTIONS,
            "setting_raw": setting_raw,
            "setting_html": setting_html,
            "lock_state": lstate,
            "char_by_id": char_by_id,
            "process_raw": process_raw,
            "best_practices_raw": bp_raw,
        }

    def _active_session_for_scenario(scen_dir: Path) -> RunSession | None:
        want = scen_dir.resolve()
        for sess in SESSIONS.values():
            if sess.finished:
                continue
            if sess.bundle.path.resolve() == want:
                return sess
        return None

    def _require_resolved_run(run_path: str) -> ResolvedRun:
        r, amb = resolve_slug_under_runs(runs_dir=runs_dir, slug=run_path)
        if amb:
            raise HTTPException(
                status_code=404,
                detail=f"Ambiguous path prefix; use full name. Matches: {', '.join(amb[:15])}",
            )
        if isinstance(r, ResolvedBatch):
            raise HTTPException(
                status_code=404,
                detail="This URL is an experiment batch, not a single run. Use /experiments/<batch_id>.",
            )
        if not isinstance(r, ResolvedRun):
            raise HTTPException(status_code=404, detail="Run not found")
        return r

    def _summary_ctx(bundle: Any) -> dict[str, Any]:
        scen = scenario_from_bundle_meta(bundle.meta)
        from harness.web.sprites import (
            character_meta_by_id,
            character_sprite_set,
            expression_from_vitals,
            sprite_url,
        )

        char_meta = character_meta_by_id(scen)
        first_world = bundle.snapshots[0].world if bundle.snapshots else {}
        last_world = bundle.snapshots[-1].world if bundle.snapshots else {}
        chars_first = (first_world.get("characters") or {}) if first_world else {}
        chars_last = (last_world.get("characters") or {}) if last_world else {}
        arc_rows: list[dict[str, Any]] = []
        for cid in sorted(set(chars_first.keys()) | set(chars_last.keys())):
            sv = chars_first.get(cid, {}).get("vitals") or {}
            ev = chars_last.get(cid, {}).get("vitals") or {}
            spark = svg_sparkline([v for _, v in vitals_history_for_character(bundle, cid, "stress")])
            ss = character_sprite_set(scen, cid)
            start_expr = expression_from_vitals(sv)
            end_expr = expression_from_vitals(ev)
            arc_rows.append(
                {
                    "id": cid,
                    "name": (char_meta.get(cid) or {}).get("name", cid),
                    "stress_spark": spark,
                    "start_vitals": sv,
                    "end_vitals": ev,
                    "start_sprite": sprite_url(ss, start_expr),
                    "end_sprite": sprite_url(ss, end_expr),
                }
            )
        metrics_delivery: list[int] = []
        metrics_happiness: list[int] = []
        for s in bundle.snapshots:
            org = (s.world or {}).get("org") or {}
            if "delivery_progress" in org:
                metrics_delivery.append(int(org.get("delivery_progress") or 0))
            if "happiness" in org:
                metrics_happiness.append(int(org.get("happiness") or 0))
        totals = (bundle.summary or {}).get("totals") or {}
        reflection_path = bundle.run_dir / "reflection.md"
        return {
            "bundle": bundle,
            "arc_rows": arc_rows,
            "metrics_delivery_svg": svg_sparkline(metrics_delivery),
            "metrics_happiness_svg": svg_sparkline(metrics_happiness),
            "summary_org": (bundle.summary or {}).get("org") or {},
            "summary_totals": totals,
            "reflection_text": (
                reflection_path.read_text(encoding="utf-8") if reflection_path.exists() else ""
            ),
        }

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return render("landing.html", request)

    @app.get("/runs", response_class=HTMLResponse)
    async def runs_index(request: Request) -> HTMLResponse:
        picker_items = build_picker_entries(runs_dir)
        return render(
            "picker.html",
            request,
            runs_dir=str(runs_dir),
            picker_items=picker_items,
        )

    @app.get("/experiments", response_class=HTMLResponse)
    async def experiments_index(request: Request) -> HTMLResponse:
        batches = list_batches(runs_dir)
        rows = []
        for p, name in batches:
            try:
                m = batch_metrics(p)
            except Exception:  # noqa: BLE001
                m = None
            rows.append({"name": name, "path": str(p), "metrics": m})
        return render("experiments_index.html", request, batches=rows)

    @app.get("/experiments/{batch_id}", response_class=HTMLResponse)
    async def experiment_detail(request: Request, batch_id: str) -> HTMLResponse:
        bd = runs_dir / batch_id
        if not bd.is_dir() or not (bd / "manifest.json").exists():
            raise HTTPException(status_code=404, detail="batch not found")
        metrics = batch_metrics(bd)
        rep = bd / "report.md"
        cmp = bd / "comparison.md"
        judge = bd / "judge_report.md"
        report_html = md_to_html(rep.read_text(encoding="utf-8")) if rep.exists() else None
        cmp_html = md_to_html(cmp.read_text(encoding="utf-8")) if cmp.exists() else None
        judge_html = md_to_html(judge.read_text(encoding="utf-8")) if judge.exists() else None
        return render(
            "experiment.html",
            request,
            batch_id=batch_id,
            metrics=metrics,
            report_html=report_html,
            comparison_html=cmp_html,
            judge_html=judge_html,
        )

    def _runner_page(
        request: Request, run_path: str, *, turn: int | None
    ) -> HTMLResponse | RedirectResponse:
        r, ambiguous = resolve_slug_under_runs(runs_dir=runs_dir, slug=run_path)
        if ambiguous:
            opts = ", ".join(ambiguous[:20])
            more = f" (+{len(ambiguous) - 20} more)" if len(ambiguous) > 20 else ""
            raise HTTPException(
                status_code=404,
                detail=(
                    f"More than one folder under runs/ starts with {run_path!r}: {opts}{more}. "
                    "Use the full directory name (for example batch_…Z). "
                    "For a batch overview, open /experiments/<that name>."
                ),
            )
        if isinstance(r, ResolvedBatch):
            return RedirectResponse(
                url=f"/experiments/{r.batch_dir.name}",
                status_code=307,
            )
        if not isinstance(r, ResolvedRun):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No run or batch at runs/{run_path!r}. "
                    "Open http://127.0.0.1:8765/ to pick a folder, or use the full batch/run id."
                ),
            )
        url_p = run_url_path(runs_dir=runs_dir, run_dir=r.run_dir)
        bundle = load_run(run_dir=r.run_dir, url_path=url_p)
        view_q = _normalize_runner_view(request.query_params.get("view"))

        live_sess: RunSession | None = SESSIONS.get(url_p)
        summ_early = bundle.summary or {}
        if live_sess is None and summ_early.get("live_session"):
            try:
                from harness.integrations.openrouter import OpenRouterClient
                from harness.runner import load_api_key

                live_sess = RunSession.from_run_dir(
                    run_root=r.run_dir,
                    client=OpenRouterClient(api_key=load_api_key(None)),
                )
                if not live_sess.finished:
                    SESSIONS[url_p] = live_sess
                else:
                    live_sess = None
            except (FileNotFoundError, OSError, ValueError, TypeError):
                live_sess = None

        active_live = live_sess is not None and not live_sess.finished

        if active_live:
            assert live_sess is not None
            max_turn = max(1, int(live_sess.world.max_turns))
            head_turn = (
                max(1, live_sess.last_completed_turn) if live_sess.last_completed_turn else 1
            )
            eff_turn = head_turn if turn is None else int(turn)
            eff_turn = max(1, min(eff_turn, max_turn, head_turn))
            snap_world = _snapshot_at(bundle, eff_turn)
            if not snap_world and live_sess.last_completed_turn == 0:
                snap_world = live_sess.world.snapshot()
            elif not snap_world:
                snap_world = live_sess.world.snapshot()
            goal_status = _goal_stoplight(bundle.summary)
            rc = _runner_ctx(bundle, snap_world, goal_status, eff_turn, max_turn)
            rc["live_turn"] = head_turn
            rc["viewing_replay"] = eff_turn != head_turn
            channel_query = request.query_params.get("channel", bundle.primary_channel)
            if channel_query not in bundle.messages_by_channel:
                channel_query = bundle.primary_channel
            msg_counts = _channel_message_counts(bundle)
            unread_counts = _channel_unread_counts(bundle, since_turn=head_turn)
            sentiment_points, sentiment_svg = _sentiment_series(bundle)
            ctx = {
                "bundle": bundle,
                "effective_turn": eff_turn,
                "max_turn": max_turn,
                "snap_world": snap_world,
                "goal_status": goal_status,
                "channel_query": channel_query,
                "current_view": view_q,
                "channel_attention": _channel_attention_map(bundle, since_turn=head_turn),
                "channel_counts": msg_counts,
                "channel_unread_counts": unread_counts,
                "vitals_sparks": _vitals_spark_map(
                    bundle, (snap_world or {}).get("characters") or {}
                ),
                "sentiment_points": sentiment_points,
                "sentiment_svg": sentiment_svg,
                **rc,
                "live_session": True,
                "session_coach_mode": live_sess.coach_mode,
                "run_url_path": url_p,
            }
        elif live_sess is not None and live_sess.finished:
            eff_turn = turn
            if eff_turn is None:
                eff_turn = int(bundle.summary.get("final_turn") or 0) or 1
            snap_world = _snapshot_at(bundle, eff_turn)
            goal_status = _goal_stoplight(bundle.summary)
            max_turn = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
            channel_query = request.query_params.get("channel", bundle.primary_channel)
            if channel_query not in bundle.messages_by_channel:
                channel_query = bundle.primary_channel
            msg_counts = _channel_message_counts(bundle)
            unread_counts = _channel_unread_counts(bundle, since_turn=eff_turn)
            sentiment_points, sentiment_svg = _sentiment_series(bundle)
            ctx = {
                "bundle": bundle,
                "effective_turn": eff_turn,
                "max_turn": max_turn,
                "snap_world": snap_world,
                "goal_status": goal_status,
                "channel_query": channel_query,
                "current_view": view_q,
                "channel_attention": _channel_attention_map(bundle, since_turn=eff_turn),
                "channel_counts": msg_counts,
                "channel_unread_counts": unread_counts,
                "vitals_sparks": _vitals_spark_map(
                    bundle, (snap_world or {}).get("characters") or {}
                ),
                "sentiment_points": sentiment_points,
                "sentiment_svg": sentiment_svg,
                **_runner_ctx(bundle, snap_world, goal_status, eff_turn, max_turn),
                "live_session": False,
                "session_coach_mode": live_sess.coach_mode,
                "run_url_path": url_p,
            }
        else:
            eff_turn = turn
            if eff_turn is None:
                eff_turn = int(bundle.summary.get("final_turn") or 0) or 1
            snap_world = _snapshot_at(bundle, eff_turn)
            goal_status = _goal_stoplight(bundle.summary)
            max_turn = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
            channel_query = request.query_params.get("channel", bundle.primary_channel)
            if channel_query not in bundle.messages_by_channel:
                channel_query = bundle.primary_channel
            msg_counts = _channel_message_counts(bundle)
            unread_counts = _channel_unread_counts(bundle, since_turn=eff_turn)
            sentiment_points, sentiment_svg = _sentiment_series(bundle)
            ctx = {
                "bundle": bundle,
                "effective_turn": eff_turn,
                "max_turn": max_turn,
                "snap_world": snap_world,
                "goal_status": goal_status,
                "channel_query": channel_query,
                "current_view": view_q,
                "channel_attention": _channel_attention_map(bundle, since_turn=eff_turn),
                "channel_counts": msg_counts,
                "channel_unread_counts": unread_counts,
                "vitals_sparks": _vitals_spark_map(
                    bundle, (snap_world or {}).get("characters") or {}
                ),
                "sentiment_points": sentiment_points,
                "sentiment_svg": sentiment_svg,
                **_runner_ctx(bundle, snap_world, goal_status, eff_turn, max_turn),
                "live_session": False,
                "session_coach_mode": (bundle.meta or {}).get("coach_mode") or "",
                "run_url_path": url_p,
            }
        if ctx.get("current_view") == "timeline":
            ctx["grouped_timeline"] = _group_timeline_rows(bundle)
            ctx["turn_ticks"] = sorted(
                {
                    int(row.raw["turn"])
                    for row in bundle.timeline
                    if row.kind == "turn_start" and row.raw.get("turn") is not None
                }
            )
            ctx["url_path"] = url_p
        snap_chars = ((ctx.get("snap_world") or {}).get("characters") or {})
        ctx["roster_work_items"] = ((ctx.get("snap_world") or {}).get("work_items") or [])
        ctx["roster_deltas"] = _roster_delta_map(
            bundle,
            snap_chars,
            int(ctx.get("effective_turn") or 1),
        )
        if ctx.get("current_view") == "summary":
            ctx.update(_summary_ctx(bundle))
        return render("runner.html", request, **ctx)

    @app.get("/new", response_class=HTMLResponse)
    async def new_run_form(request: Request) -> HTMLResponse:
        cards = _scenario_cards(scenarios_root)
        selected = request.query_params.get("scenario", "").strip()
        return render(
            "new_run.html",
            request,
            scenarios=cards,
            selected_scenario=selected,
            runs_dir=str(runs_dir),
        )

    @app.post("/runs")
    async def start_live_run(
        request: Request,
        scenario_slug: str = Form(...),
        agent_model: str = Form("google/gemini-3-flash-preview"),
        coach_model: str = Form(""),
        coach_mode: str = Form("llm"),
    ) -> RedirectResponse:
        scen = _safe_scenario_dir(scenarios_root, scenario_slug)
        try:
            from harness.integrations.openrouter import OpenRouterClient

            client = OpenRouterClient(api_key=load_api_key(None))
        except (FileNotFoundError, ValueError, OSError) as err:
            raise HTTPException(
                status_code=500,
                detail="Need secrets.yaml with openrouter_api_key to start a live run.",
            ) from err
        cm = coach_model.strip() or None
        sess = RunSession.start(
            scenario_dir=scen,
            runs_dir=runs_dir,
            agent_model=agent_model.strip() or "google/gemini-3-flash-preview",
            coach_model=cm,
            coach_mode_cli=coach_mode.strip().lower(),
            coach_preset_cli=None,
            secrets=None,
            client=client,
            seed=None,
        )
        url_p = run_url_path(runs_dir=runs_dir, run_dir=sess.run_root)
        SESSIONS[url_p] = sess
        return RedirectResponse(url=f"/runs/{url_p}", status_code=303)

    @app.post("/runs/{run_path}/advance", response_class=HTMLResponse)
    async def live_advance(request: Request, run_path: str) -> HTMLResponse:
        r = _require_resolved_run(run_path)
        url_p = run_url_path(runs_dir=runs_dir, run_dir=r.run_dir)
        sess = SESSIONS.get(url_p)
        if sess is None:
            raise HTTPException(status_code=404, detail="no active live session")
        await sess.advance()
        bundle = load_run(run_dir=r.run_dir, url_path=url_p)
        channel = bundle.primary_channel
        live_sess = SESSIONS.get(url_p)
        active = live_sess is not None and not live_sess.finished
        head_turn = (
            max(1, live_sess.last_completed_turn) if live_sess and live_sess.last_completed_turn else 1
        )
        if live_sess and live_sess.last_completed_turn == 0:
            head_turn = 1
        max_t = live_sess.world.max_turns if live_sess else 1
        snap = _snapshot_at(bundle, head_turn) or (
            live_sess.world.snapshot()
            if live_sess and live_sess.last_completed_turn == 0
            else {}
        )
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, snap, gs, head_turn, max_t)
        sparks = _vitals_spark_map(bundle, (snap or {}).get("characters") or {})
        goals_html = jinja.get_template("partials/goals_panel.html").render(
            request=request,
            bundle=bundle,
            **rc,
            goal_status=gs,
            effective_turn=head_turn,
        )
        vitals_html = jinja.get_template("partials/vitals_rail.html").render(
            request=request,
            bundle=bundle,
            **rc,
            characters=(snap or {}).get("characters") or {},
            effective_turn=head_turn,
            vitals_sparks=sparks,
        )
        msgs = bundle.messages_by_channel.get(channel, [])
        channel_html = jinja.get_template("partials/channel_view.html").render(
            request=request,
            bundle=bundle,
            **rc,
            channel=channel,
            messages=msgs,
            effective_turn=head_turn,
            snap_world=snap,
            live_session=active,
            session_coach_mode=live_sess.coach_mode if live_sess else "",
        )
        oob = (
            f'<div id="goals-panel" hx-swap-oob="true">{goals_html}</div>'
            f'<div id="vitals-rail" hx-swap-oob="true">{vitals_html}</div>'
            f'<div id="center-stage" hx-swap-oob="true">{channel_html}</div>'
        )
        return HTMLResponse(oob)

    @app.get("/runs/{run_path}/events")
    async def live_events(run_path: str) -> StreamingResponse:
        r = _require_resolved_run(run_path)
        url_p = run_url_path(runs_dir=runs_dir, run_dir=r.run_dir)
        sess = SESSIONS.get(url_p)
        if sess is None:
            raise HTTPException(status_code=404, detail="no active live session")

        async def _gen():
            while True:
                try:
                    evt = await asyncio.wait_for(sess.events.get(), timeout=5.0)
                except TimeoutError:
                    yield "event: progress\ndata: {}\n\n"
                    break
                yield f"event: progress\ndata: {json.dumps(evt)}\n\n"
                if evt.get("kind") in {"advance_stop", "advance_cancelled"}:
                    break

        return StreamingResponse(_gen(), media_type="text/event-stream")

    @app.post("/runs/{run_path}/cancel")
    async def live_cancel(run_path: str) -> dict[str, Any]:
        r = _require_resolved_run(run_path)
        url_p = run_url_path(runs_dir=runs_dir, run_dir=r.run_dir)
        sess = SESSIONS.get(url_p)
        if sess is None:
            raise HTTPException(status_code=404, detail="no active live session")
        sess.cancel()
        return {"ok": True}

    @app.post("/runs/{run_path}/coach/post", response_class=HTMLResponse)
    async def live_coach_post(
        request: Request,
        run_path: str,
        channel: str = Form(...),
        content: str = Form(...),
    ) -> HTMLResponse:
        r = _require_resolved_run(run_path)
        url_p = run_url_path(runs_dir=runs_dir, run_dir=r.run_dir)
        sess = SESSIONS.get(url_p)
        if sess is None or sess.finished:
            raise HTTPException(status_code=400, detail="no active live session")
        sess.coach_post(channel=channel, content=content, author="coach")
        bundle = load_run(run_dir=r.run_dir, url_path=url_p)
        ch = channel.strip() or bundle.primary_channel
        head_turn = max(1, sess.last_completed_turn) if sess.last_completed_turn else 1
        max_t = sess.world.max_turns
        snap = _snapshot_at(bundle, head_turn) or sess.world.snapshot()
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, snap, gs, head_turn, max_t)
        msgs = bundle.messages_by_channel.get(ch, [])
        channel_html = jinja.get_template("partials/channel_view.html").render(
            request=request,
            bundle=bundle,
            **rc,
            channel=ch,
            messages=msgs,
            effective_turn=head_turn,
            snap_world=snap,
            live_session=True,
            session_coach_mode=sess.coach_mode,
        )
        return HTMLResponse(channel_html)

    @app.post("/runs/{run_path}/reflection", response_class=HTMLResponse)
    async def live_reflection(
        request: Request, run_path: str, content: str = Form("")
    ) -> HTMLResponse:
        r = _require_resolved_run(run_path)
        url_p = run_url_path(runs_dir=runs_dir, run_dir=r.run_dir)
        sess = SESSIONS.get(url_p)
        if sess is None:
            raise HTTPException(status_code=404, detail="no active live session")
        sess.write_reflection(content)
        return HTMLResponse('<div class="text-xs text-[var(--color-text-muted)]">Reflection saved.</div>')

    @app.post("/runs/{run_path}/edit/vital")
    async def live_edit_vital(
        run_path: str,
        character_id: str = Form(...),
        vital_name: str = Form(...),
        delta: int = Form(...),
    ) -> dict[str, Any]:
        r = _require_resolved_run(run_path)
        url_p = run_url_path(runs_dir=runs_dir, run_dir=r.run_dir)
        sess = SESSIONS.get(url_p)
        if sess is None:
            raise HTTPException(status_code=404, detail="no active live session")
        try:
            info = sess.edit_vital(character_id=character_id, vital_name=vital_name, delta=delta)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        return {"ok": True, "message": "applied — takes effect at turn N+1", "edit": info}

    @app.post("/runs/{run_path}/edit/parameter")
    async def live_edit_parameter(
        run_path: str,
        key: str = Form(...),
        value: str = Form(...),
    ) -> dict[str, Any]:
        r = _require_resolved_run(run_path)
        url_p = run_url_path(runs_dir=runs_dir, run_dir=r.run_dir)
        sess = SESSIONS.get(url_p)
        if sess is None:
            raise HTTPException(status_code=404, detail="no active live session")
        info = sess.edit_parameter(key=key, value=value)
        return {"ok": True, "message": "applied — takes effect at turn N+1", "edit": info}

    @app.get("/partials/run/{run_path:path}/kanban", response_class=HTMLResponse)
    async def partial_kanban(request: Request, run_path: str) -> HTMLResponse:
        turn_s = request.query_params.get("turn")
        turn = int(turn_s) if turn_s and turn_s.isdigit() else None
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        world = _snapshot_at(bundle, turn) or {}
        items = world.get("work_items") or []
        cols = {"backlog": [], "doing": [], "done": [], "parked": []}
        for wi in items:
            st = str(wi.get("state", "backlog")).lower()
            if st in cols:
                cols[st].append(wi)
        eff = turn if turn is not None else max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        max_t = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, world, gs, eff, max_t)
        ctx = {**rc, "bundle": bundle, "columns": cols, "effective_turn": eff, "snap_world": world}
        return render("partials/kanban.html", request, **ctx)

    @app.get("/partials/run/{run_path:path}/timeline", response_class=HTMLResponse)
    async def partial_timeline(request: Request, run_path: str) -> HTMLResponse:
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        turn_s = request.query_params.get("turn")
        turn = int(turn_s) if turn_s and turn_s.isdigit() else None
        eff = turn if turn is not None else max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        turn_ticks = sorted(
            {
                int(row.raw["turn"])
                for row in bundle.timeline
                if row.kind == "turn_start" and row.raw.get("turn") is not None
            }
        )
        grouped_timeline = _group_timeline_rows(bundle)
        return render(
            "partials/timeline.html",
            request,
            bundle=bundle,
            url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir),
            turn_ticks=turn_ticks,
            grouped_timeline=grouped_timeline,
            effective_turn=eff,
        )

    @app.get("/partials/run/{run_path:path}/summary", response_class=HTMLResponse)
    async def partial_summary(request: Request, run_path: str) -> HTMLResponse:
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        return render("partials/summary.html", request, **_summary_ctx(bundle))

    @app.post("/runs/{run_path:path}/generate_report", response_class=HTMLResponse)
    async def run_generate_report(request: Request, run_path: str) -> HTMLResponse:
        r = _require_resolved_run(run_path)
        analyse_batch(r.run_dir)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        return render("partials/summary.html", request, **_summary_ctx(bundle))

    @app.get("/partials/run/{run_path:path}/vitals", response_class=HTMLResponse)
    async def partial_vitals(request: Request, run_path: str) -> HTMLResponse:
        turn_s = request.query_params.get("turn")
        turn = int(turn_s) if turn_s and turn_s.isdigit() else None
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        world = _snapshot_at(bundle, turn) or {}
        chars = world.get("characters") or {}
        eff = turn if turn is not None else max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        max_t = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, world, gs, eff, max_t)
        sparks = _vitals_spark_map(bundle, chars)
        sentiment_points, sentiment_svg = _sentiment_series(bundle)
        ctx = {
            **rc,
            "bundle": bundle,
            "characters": chars,
            "effective_turn": eff,
            "vitals_sparks": sparks,
            "sentiment_points": sentiment_points,
            "sentiment_svg": sentiment_svg,
        }
        return render("partials/vitals_rail.html", request, **ctx)

    @app.get("/partials/run/{run_path:path}/goals", response_class=HTMLResponse)
    async def partial_goals(request: Request, run_path: str) -> HTMLResponse:
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        turn_s = request.query_params.get("turn")
        turn = int(turn_s) if turn_s and turn_s.isdigit() else None
        world = _snapshot_at(bundle, turn) or {}
        eff = turn if turn is not None else max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        max_t = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, world, gs, eff, max_t)
        url_px = run_url_path(runs_dir=runs_dir, run_dir=r.run_dir)
        ls = SESSIONS.get(url_px)
        ctx = {
            **rc,
            "bundle": bundle,
            "goal_status": gs,
            "effective_turn": eff,
            "live_session": bool(ls and not ls.finished),
        }
        return render("partials/goals_panel.html", request, **ctx)

    @app.get("/partials/run/{run_path:path}/roster", response_class=HTMLResponse)
    async def partial_roster(request: Request, run_path: str) -> HTMLResponse:
        turn_s = request.query_params.get("turn")
        turn = int(turn_s) if turn_s and turn_s.isdigit() else None
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        world = _snapshot_at(bundle, turn) or {}
        chars = world.get("characters") or {}
        eff = turn if turn is not None else max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        max_t = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, world, gs, eff, max_t)
        sparks = _vitals_spark_map(bundle, chars)
        deltas = _roster_delta_map(bundle, chars, eff)
        ctx = {
            **rc,
            "bundle": bundle,
            "characters": chars,
            "effective_turn": eff,
            "vitals_sparks": sparks,
            "roster_deltas": deltas,
            "roster_work_items": (world.get("work_items") or []),
        }
        return render("partials/roster.html", request, **ctx)

    @app.get(
        "/partials/run/{run_path:path}/inspector/character/{cid}",
        response_class=HTMLResponse,
    )
    async def partial_inspector_character(request: Request, run_path: str, cid: str) -> HTMLResponse:
        turn_s = request.query_params.get("turn")
        turn = int(turn_s) if turn_s and turn_s.isdigit() else None
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        world = _snapshot_at(bundle, turn) or {}
        eff = turn if turn is not None else max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        max_t = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, world, gs, eff, max_t)
        ch = (world.get("characters") or {}).get(cid, {})
        narr = agent_narrative_snippets(bundle, cid)
        posts = recent_posts_for_character(bundle, cid)
        ctx = {
            **rc,
            "bundle": bundle,
            "inspect_cid": cid,
            "inspect_character": ch,
            "inspect_narratives": narr,
            "inspect_recent_posts": posts,
            "effective_turn": eff,
        }
        return render("partials/inspector_character.html", request, **ctx)

    @app.get(
        "/partials/run/{run_path:path}/inspector/work_item/{wid}",
        response_class=HTMLResponse,
    )
    async def partial_inspector_work_item(request: Request, run_path: str, wid: str) -> HTMLResponse:
        turn_s = request.query_params.get("turn")
        turn = int(turn_s) if turn_s and turn_s.isdigit() else None
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        world = _snapshot_at(bundle, turn) or {}
        eff = turn if turn is not None else max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        max_t = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, world, gs, eff, max_t)
        wi = None
        for w in world.get("work_items") or []:
            if str(w.get("id")) == str(wid):
                wi = w
                break
        ev = work_item_timeline_events(bundle, wid)
        ctx = {
            **rc,
            "bundle": bundle,
            "inspect_wi": wi,
            "inspect_wi_id": wid,
            "inspect_wi_events": ev,
            "effective_turn": eff,
        }
        return render("partials/inspector_work_item.html", request, **ctx)

    @app.get("/partials/run/{run_path:path}/inspector/channel", response_class=HTMLResponse)
    async def partial_inspector_channel(request: Request, run_path: str) -> HTMLResponse:
        ch = request.query_params.get("channel", "")
        turn_s = request.query_params.get("turn")
        turn = int(turn_s) if turn_s and turn_s.isdigit() else None
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        world = _snapshot_at(bundle, turn) or {}
        eff = turn if turn is not None else max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        max_t = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, world, gs, eff, max_t)
        scen = rc.get("scenario") or {}
        meta = channel_meta_for(bundle, ch, scen)
        ctx = {**rc, "bundle": bundle, "inspect_channel": ch, "channel_inspector_meta": meta, "effective_turn": eff}
        return render("partials/inspector_channel.html", request, **ctx)

    @app.get("/runs/{run_path:path}/inspect/{object_type}/{object_id:path}", response_class=HTMLResponse)
    async def runner_inspect_full(
        request: Request,
        run_path: str,
        object_type: str,
        object_id: str,
    ) -> HTMLResponse:
        turn_s = request.query_params.get("turn")
        turn = int(turn_s) if turn_s and turn_s.isdigit() else None
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        world = _snapshot_at(bundle, turn) or {}
        eff = turn if turn is not None else max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        max_t = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, world, gs, eff, max_t)
        kind = str(object_type or "").strip().lower()
        ctx: dict[str, Any] = {
            **rc,
            "bundle": bundle,
            "effective_turn": eff,
            "full_page": True,
            "inspect_partial": "",
            "inspect_title": "",
        }
        if kind == "character":
            cid = str(object_id).strip()
            ch = (world.get("characters") or {}).get(cid, {})
            ctx.update(
                {
                    "inspect_partial": "partials/inspector_character.html",
                    "inspect_title": f"Character {cid}",
                    "inspect_cid": cid,
                    "inspect_character": ch,
                    "inspect_narratives": agent_narrative_snippets(bundle, cid),
                    "inspect_recent_posts": recent_posts_for_character(bundle, cid),
                }
            )
        elif kind == "work_item":
            wid = str(object_id).strip()
            wi = None
            for w in world.get("work_items") or []:
                if str(w.get("id")) == wid:
                    wi = w
                    break
            ctx.update(
                {
                    "inspect_partial": "partials/inspector_work_item.html",
                    "inspect_title": f"Work item {wid}",
                    "inspect_wi": wi,
                    "inspect_wi_id": wid,
                    "inspect_wi_events": work_item_timeline_events(bundle, wid),
                }
            )
        elif kind == "channel":
            ch_name = str(object_id).strip()
            meta = channel_meta_for(bundle, ch_name, rc.get("scenario") or {})
            ctx.update(
                {
                    "inspect_partial": "partials/inspector_channel.html",
                    "inspect_title": f"Channel {ch_name}",
                    "inspect_channel": ch_name,
                    "channel_inspector_meta": meta,
                }
            )
        else:
            raise HTTPException(status_code=404, detail="unknown inspect object")
        return render("inspect_full.html", request, **ctx)

    @app.get("/partials/run/{run_path:path}/channel", response_class=HTMLResponse)
    async def partial_channel(request: Request, run_path: str) -> HTMLResponse:
        ch = request.query_params.get("channel", "")
        turn_s = request.query_params.get("turn")
        turn = int(turn_s) if turn_s and turn_s.isdigit() else None
        r = _require_resolved_run(run_path)
        bundle = load_run(run_dir=r.run_dir, url_path=run_url_path(runs_dir=runs_dir, run_dir=r.run_dir))
        if not ch:
            ch = bundle.primary_channel
        msgs = bundle.messages_by_channel.get(ch, [])
        eff = turn if turn is not None else max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        max_t = max(1, int(bundle.summary.get("final_turn") or 0) or 1)
        snap = _snapshot_at(bundle, turn) or {}
        gs = _goal_stoplight(bundle.summary)
        rc = _runner_ctx(bundle, snap, gs, eff, max_t)
        url_px = run_url_path(runs_dir=runs_dir, run_dir=r.run_dir)
        ls = SESSIONS.get(url_px)
        active_live = ls is not None and not ls.finished
        ctx = {
            **rc,
            "bundle": bundle,
            "channel": ch,
            "messages": msgs,
            "effective_turn": eff,
            "snap_world": snap,
            "live_session": active_live,
            "session_coach_mode": ls.coach_mode if ls else "",
        }
        return render("partials/channel_view.html", request, **ctx)

    @app.get("/scenarios", response_class=HTMLResponse)
    async def scenarios_index(request: Request) -> HTMLResponse:
        cards = _scenario_cards(scenarios_root)
        return render("scenarios_index.html", request, scenarios=cards)

    @app.get("/scenarios/new", response_class=HTMLResponse)
    async def scenarios_new_form(request: Request) -> HTMLResponse:
        cards = _scenario_cards(scenarios_root)
        return render("scenario_new.html", request, scenarios=cards, error="", created_slug="")

    @app.post("/scenarios/new", response_class=HTMLResponse)
    async def scenarios_new_create(
        request: Request,
        scenario_id: str = Form(""),
        name: str = Form(""),
        description: str = Form(""),
    ) -> HTMLResponse:
        cards = _scenario_cards(scenarios_root)
        raw_id = str(scenario_id or "").strip()
        raw_name = str(name or "").strip()
        sid = _slugify(raw_id or raw_name)
        if not sid:
            return render(
                "scenario_new.html",
                request,
                scenarios=cards,
                error="Scenario id or name is required.",
                created_slug="",
            )
        target = scenarios_root / sid
        if target.exists():
            return render(
                "scenario_new.html",
                request,
                scenarios=cards,
                error=f"Scenario '{sid}' already exists.",
                created_slug="",
            )
        target.mkdir(parents=True, exist_ok=False)
        scenario_data = {
            "id": sid,
            "name": raw_name or sid.replace("-", " ").title(),
            "description": description.strip(),
            "setting_file": "setting.md",
            "best_practices_file": "best_practices.yaml",
            "channels": [{"name": "#team", "type": "open", "coach_engagement": "post", "member_ids": ["lead"]}],
            "teams": [{"id": "core", "name": "Core", "member_ids": ["lead"]}],
            "characters": [
                {
                    "id": "lead",
                    "name": "Team Lead",
                    "role": "Lead",
                    "sprite_set": "char_lead",
                    "markdown_file": "characters/lead.md",
                    "initial_vitals": {"energy": 60, "motivation": 60, "stress": 40},
                }
            ],
            "work_items": [],
            "goals": {"max_turns": 12, "max_stress_any": 90, "abort_stress_any": 100},
            "parameters": {},
        }
        (target / "characters").mkdir(parents=True, exist_ok=True)
        (target / "scenario.yaml").write_text(
            yaml.safe_dump(scenario_data, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        (target / "setting.md").write_text(
            f"# {scenario_data['name']}\n\nDescribe the situation and constraints.\n",
            encoding="utf-8",
        )
        (target / "characters" / "lead.md").write_text(
            "Team lead profile and coaching context.\n",
            encoding="utf-8",
        )
        (target / "best_practices.yaml").write_text("practices: []\n", encoding="utf-8")
        (target / "process.yaml").write_text("rules: []\n", encoding="utf-8")
        return RedirectResponse(url=f"/scenarios/{sid}/edit", status_code=303)

    @app.get("/scenarios/{slug}", response_class=HTMLResponse)
    async def scenarios_view(request: Request, slug: str) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        data = json.loads(json.dumps(yaml.safe_load((scen_dir / "scenario.yaml").read_text(encoding="utf-8")) or {}))
        setting_md = (scen_dir / str(data.get("setting_file") or "setting.md"))
        setting_html = md_to_html(setting_md.read_text(encoding="utf-8")) if setting_md.exists() else ""
        return render(
            "scenario_view.html",
            request,
            slug=slug,
            scenario=data,
            setting_html=setting_html,
        )

    @app.get("/scenarios/{slug}/edit", response_class=HTMLResponse)
    async def scenarios_edit(request: Request, slug: str) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        section = str(request.query_params.get("section") or "setting").strip().lower()
        if section not in _EDITOR_SECTIONS:
            section = "setting"
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section=section)
        selected_char = str(request.query_params.get("character") or "").strip()
        if selected_char:
            ctx["selected_character_id"] = selected_char
        return render(
            "scenario_editor.html",
            request,
            **ctx,
            scenarios_dir=str(scenarios_root),
        )

    @app.get("/partials/scenario/{slug}/section/{section}", response_class=HTMLResponse)
    async def scenario_partial_section(request: Request, slug: str, section: str) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sec = str(section).strip().lower()
        if sec not in _EDITOR_SECTIONS:
            raise HTTPException(status_code=404, detail="unknown section")
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section=sec)
        selected_char = str(request.query_params.get("character") or "").strip()
        if selected_char:
            ctx["selected_character_id"] = selected_char
        if sec == "process":
            ctx["process_rows"] = _yaml_line_rows(str(ctx.get("process_raw") or ""), None)
            ctx["error_line"] = None
        if sec == "best_practices":
            raw = str(ctx.get("best_practices_raw") or "")
            ctx["best_practices_rows"] = _yaml_line_rows(raw, None)
            ctx["error_line"] = None
        tpl_name = f"partials/scenario_editor/section_{sec}.html"
        return render(tpl_name, request, **ctx)

    @app.post("/scenarios/{slug}/edit/setting", response_class=HTMLResponse)
    async def scenario_edit_setting(
        request: Request,
        slug: str,
        content: str = Form(""),
        preview: str = Form("0"),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        write_setting_md(scen_dir, content)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="setting")
        preview_on = str(preview).strip() in {"1", "true", "yes", "on"}
        ctx.update(
            {
                "setting_raw": content,
                "setting_html": md_to_html(content) if content else "",
                "saved": True,
                "preview": preview_on,
            }
        )
        return render(
            "partials/scenario_editor/section_setting.html",
            request,
            **ctx,
        )

    @app.get("/partials/scenario/{slug}/inspector/character/{cid}", response_class=HTMLResponse)
    async def scenario_inspector_character(request: Request, slug: str, cid: str) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="characters")
        char_id = validate_character_id(cid)
        ch = (ctx.get("char_by_id") or {}).get(char_id)
        if not ch:
            raise HTTPException(status_code=404, detail="character not found")
        md_path = character_md_path(scen_dir, ch)
        backstory = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        return render(
            "partials/scenario_editor/inspector_character.html",
            request,
            **ctx,
            inspect_character=ch,
            inspect_cid=char_id,
            backstory_raw=backstory,
            backstory_html=md_to_html(backstory) if backstory else "",
            backstory_shared=is_shared_character_path(scen_dir, md_path),
            saved=False,
            preview=False,
            message="",
            error="",
        )

    @app.post("/scenarios/{slug}/edit/character/{cid}", response_class=HTMLResponse)
    async def scenario_edit_character(
        request: Request,
        slug: str,
        cid: str,
        name: str = Form(""),
        role: str = Form(""),
        sprite_set: str = Form(""),
        model: str = Form(""),
        energy: str = Form(""),
        motivation: str = Form(""),
        stress: str = Form(""),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        char_id = validate_character_id(cid)
        err = ""
        msg = ""
        sess = _active_session_for_scenario(scen_dir)
        current = _scenario_editor_ctx(scen_dir, slug=slug, section="characters").get("char_by_id", {}).get(char_id)
        if sess is not None and current:
            iv = current.get("initial_vitals") or {}
            def _num_changed(raw: str, old_v: Any) -> bool:
                txt = str(raw).strip()
                if txt == "":
                    return False
                try:
                    return int(txt) != int(old_v)
                except (TypeError, ValueError):
                    return True
            locked_changed = (
                (str(name).strip() and str(name).strip() != str(current.get("name") or ""))
                or (str(role).strip() and str(role).strip() != str(current.get("role") or ""))
                or _num_changed(energy, iv.get("energy", 0))
                or _num_changed(motivation, iv.get("motivation", 0))
                or _num_changed(stress, iv.get("stress", 0))
            )
            if locked_changed:
                err = "locked: character core profile and starting vitals cannot change during a live run"
        try:
            if not err:
                upsert_character(
                    scen_dir,
                    char_id=char_id,
                    fields={
                        "name": name,
                        "role": role,
                        "sprite_set": sprite_set,
                        "model": model,
                        "energy": energy,
                        "motivation": motivation,
                        "stress": stress,
                    },
                )
                msg = "Saved character fields."
                if sess is not None:
                    sess.record_scenario_edit(
                        target="character",
                        payload={"character_id": char_id, "fields": {"sprite_set": sprite_set, "model": model}},
                    )
        except ValueError as e:
            err = str(e)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="characters")
        ch = (ctx.get("char_by_id") or {}).get(char_id)
        if not ch:
            raise HTTPException(status_code=404, detail="character not found")
        md_path = character_md_path(scen_dir, ch)
        backstory = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        resp = render(
            "partials/scenario_editor/inspector_character.html",
            request,
            **ctx,
            inspect_character=ch,
            inspect_cid=char_id,
            backstory_raw=backstory,
            backstory_html=md_to_html(backstory) if backstory else "",
            backstory_shared=is_shared_character_path(scen_dir, md_path),
            saved=not bool(err),
            preview=False,
            message=msg,
            error=err,
        )
        if err.startswith("locked:"):
            resp.status_code = 423
        return resp

    @app.post("/scenarios/{slug}/edit/character/{cid}/backstory", response_class=HTMLResponse)
    async def scenario_edit_character_backstory(
        request: Request,
        slug: str,
        cid: str,
        content: str = Form(""),
        preview: str = Form("0"),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        char_id = validate_character_id(cid)
        err = ""
        msg = ""
        sess = _active_session_for_scenario(scen_dir)
        if sess is not None:
            err = "locked: character backstory cannot change during a live run"
        try:
            if not err:
                write_character_backstory(scen_dir, char_id=char_id, content=content)
                msg = "Saved backstory."
        except ValueError as e:
            err = str(e)
        preview_on = str(preview).strip() in {"1", "true", "yes", "on"}
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="characters")
        ch = (ctx.get("char_by_id") or {}).get(char_id)
        if not ch:
            raise HTTPException(status_code=404, detail="character not found")
        resp = render(
            "partials/scenario_editor/inspector_character.html",
            request,
            **ctx,
            inspect_character=ch,
            inspect_cid=char_id,
            backstory_raw=content,
            backstory_html=md_to_html(content) if content else "",
            backstory_shared=is_shared_character_path(scen_dir, character_md_path(scen_dir, ch)),
            saved=not bool(err),
            preview=preview_on,
            message=msg,
            error=err,
        )
        if err.startswith("locked:"):
            resp.status_code = 423
        return resp

    @app.post("/scenarios/{slug}/character/new", response_class=HTMLResponse)
    async def scenario_character_new(
        request: Request,
        slug: str,
        id: str = Form(...),
        name: str = Form(""),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        err = ""
        if _active_session_for_scenario(scen_dir) is not None:
            err = "locked: cannot add characters during a live run"
        try:
            if not err:
                create_character(scen_dir, char_id=id, name=name)
        except ValueError as e:
            err = str(e)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="characters")
        resp = render(
            "partials/scenario_editor/section_characters.html",
            request,
            **ctx,
            selected_character_id=str(id).strip(),
            message="" if err else "Character created.",
            error=err,
        )
        if err.startswith("locked:"):
            resp.status_code = 423
        return resp

    @app.post("/scenarios/{slug}/character/{cid}/delete", response_class=HTMLResponse)
    async def scenario_character_delete(
        request: Request,
        slug: str,
        cid: str,
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        err = ""
        msg = ""
        if _active_session_for_scenario(scen_dir) is not None:
            err = "locked: cannot delete characters during a live run"
        try:
            if not err:
                delete_character(scen_dir, char_id=cid)
                msg = "Character deleted."
        except ValueError as e:
            err = str(e)
        except RuntimeError as e:
            err = str(e)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="characters")
        resp = render(
            "partials/scenario_editor/section_characters.html",
            request,
            **ctx,
            selected_character_id="",
            message=msg,
            error=err,
        )
        if err.startswith("locked:"):
            resp.status_code = 423
        return resp

    @app.post("/scenarios/{slug}/edit/channel", response_class=HTMLResponse)
    async def scenario_edit_channel(
        request: Request,
        slug: str,
        name: str = Form(""),
        type: str = Form("open"),
        coach_engagement: str = Form("post"),
        member_ids: list[str] = Form(default=[]),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sess = _active_session_for_scenario(scen_dir)
        err = ""
        msg = ""
        try:
            upsert_channel(
                scen_dir,
                name=name,
                channel_type=type,
                coach_engagement=coach_engagement,
                member_ids=member_ids,
            )
            msg = "Channel saved."
            if sess is not None:
                sess.record_scenario_edit(
                    target="channel",
                    payload={"name": name, "coach_engagement": coach_engagement, "member_ids": member_ids},
                )
        except ValueError as e:
            err = str(e)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="channels")
        return render("partials/scenario_editor/section_channels.html", request, **ctx, message=msg, error=err)

    @app.post("/scenarios/{slug}/channel/delete", response_class=HTMLResponse)
    async def scenario_channel_delete(
        request: Request,
        slug: str,
        name: str = Form(""),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sess = _active_session_for_scenario(scen_dir)
        delete_channel(scen_dir, name=name)
        if sess is not None:
            sess.record_scenario_edit(target="channel_delete", payload={"name": name})
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="channels")
        return render(
            "partials/scenario_editor/section_channels.html",
            request,
            **ctx,
            message="Channel deleted.",
            error="",
        )

    @app.post("/scenarios/{slug}/edit/team", response_class=HTMLResponse)
    async def scenario_edit_team(
        request: Request,
        slug: str,
        id: str = Form(""),
        name: str = Form(""),
        member_ids: list[str] = Form(default=[]),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sess = _active_session_for_scenario(scen_dir)
        err = ""
        msg = ""
        try:
            upsert_team(scen_dir, team_id=id, name=name, member_ids=member_ids)
            msg = "Team saved."
            if sess is not None:
                sess.record_scenario_edit(
                    target="team",
                    payload={"id": id, "member_ids": member_ids},
                )
        except ValueError as e:
            err = str(e)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="teams")
        return render("partials/scenario_editor/section_teams.html", request, **ctx, message=msg, error=err)

    @app.post("/scenarios/{slug}/team/delete", response_class=HTMLResponse)
    async def scenario_team_delete(
        request: Request,
        slug: str,
        id: str = Form(""),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sess = _active_session_for_scenario(scen_dir)
        delete_team(scen_dir, team_id=id)
        if sess is not None:
            sess.record_scenario_edit(target="team_delete", payload={"id": id})
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="teams")
        return render(
            "partials/scenario_editor/section_teams.html",
            request,
            **ctx,
            message="Team deleted.",
            error="",
        )

    @app.post("/scenarios/{slug}/edit/work_item", response_class=HTMLResponse)
    async def scenario_edit_work_item(
        request: Request,
        slug: str,
        id: str = Form(""),
        title: str = Form(""),
        state: str = Form("backlog"),
        owner_id: str = Form(""),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sess = _active_session_for_scenario(scen_dir)
        if sess is not None:
            ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="work_items")
            resp = render(
                "partials/scenario_editor/section_work_items.html",
                request,
                **ctx,
                message="",
                error="locked: work items are initial conditions and cannot change during a live run",
            )
            resp.status_code = 423
            return resp
        err = ""
        msg = ""
        try:
            upsert_work_item(
                scen_dir,
                work_id=id,
                title=title,
                state=state,
                owner_id=owner_id,
            )
            msg = "Work item saved."
        except ValueError as e:
            err = str(e)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="work_items")
        return render("partials/scenario_editor/section_work_items.html", request, **ctx, message=msg, error=err)

    @app.post("/scenarios/{slug}/work_item/delete", response_class=HTMLResponse)
    async def scenario_work_item_delete(
        request: Request,
        slug: str,
        id: str = Form(""),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sess = _active_session_for_scenario(scen_dir)
        if sess is not None:
            ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="work_items")
            resp = render(
                "partials/scenario_editor/section_work_items.html",
                request,
                **ctx,
                message="",
                error="locked: work items are initial conditions and cannot change during a live run",
            )
            resp.status_code = 423
            return resp
        delete_work_item(scen_dir, work_id=id)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="work_items")
        return render(
            "partials/scenario_editor/section_work_items.html",
            request,
            **ctx,
            message="Work item deleted.",
            error="",
        )

    @app.post("/scenarios/{slug}/edit/goals", response_class=HTMLResponse)
    async def scenario_edit_goals(
        request: Request,
        slug: str,
        max_turns: str = Form(""),
        max_stress_any: str = Form(""),
        abort_stress_any: str = Form(""),
        min_done_work_items: str = Form(""),
        per_team_min_done: str = Form(""),
        require_done_ids: list[str] = Form(default=[]),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sess = _active_session_for_scenario(scen_dir)
        err = ""
        msg = ""
        try:
            update_goals(
                scen_dir,
                max_turns=max_turns,
                max_stress_any=max_stress_any,
                abort_stress_any=abort_stress_any,
                min_done_work_items=min_done_work_items,
                per_team_min_done=per_team_min_done,
                require_done_ids=require_done_ids,
            )
            msg = "Goals saved."
            if sess is not None:
                sess.record_scenario_edit(
                    target="goals",
                    payload={"require_done_ids": require_done_ids},
                )
        except ValueError as e:
            err = str(e)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="goals")
        return render("partials/scenario_editor/section_goals.html", request, **ctx, message=msg, error=err)

    @app.post("/scenarios/{slug}/edit/parameters", response_class=HTMLResponse)
    async def scenario_edit_parameters(
        request: Request,
        slug: str,
        param_keys: list[str] = Form(default=[]),
        param_values: list[str] = Form(default=[]),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sess = _active_session_for_scenario(scen_dir)
        err = ""
        msg = ""
        try:
            update_parameters(scen_dir, keys=param_keys, values=param_values)
            msg = "Parameters saved."
            if sess is not None:
                sess.record_scenario_edit(
                    target="parameters",
                    payload={"keys": param_keys},
                )
        except ValueError as e:
            err = str(e)
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="parameters")
        return render(
            "partials/scenario_editor/section_parameters.html",
            request,
            **ctx,
            message=msg,
            error=err,
        )

    @app.post("/scenarios/{slug}/edit/process", response_class=HTMLResponse)
    async def scenario_edit_process(
        request: Request,
        slug: str,
        content: str = Form(""),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sess = _active_session_for_scenario(scen_dir)
        result = write_yaml_file_safely(process_yaml_path(scen_dir), content)
        if result.ok and sess is not None:
            sess.record_scenario_edit(target="process", payload={"chars": len(content or "")})
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="process")
        process_raw = content
        if result.ok:
            process_raw = read_yaml_text(process_yaml_path(scen_dir))
        ctx.update(
            {
                "process_raw": process_raw,
                "process_rows": _yaml_line_rows(process_raw, result.error_line),
                "error": result.error_message if not result.ok else "",
                "message": "Process YAML saved." if result.ok else "",
                "error_line": result.error_line,
            }
        )
        resp = render("partials/scenario_editor/section_process.html", request, **ctx)
        if not result.ok:
            resp.status_code = 422
        return resp

    @app.post("/scenarios/{slug}/edit/best_practices", response_class=HTMLResponse)
    async def scenario_edit_best_practices(
        request: Request,
        slug: str,
        content: str = Form(""),
    ) -> HTMLResponse:
        scen_dir = _safe_scenario_dir(scenarios_root, slug)
        sess = _active_session_for_scenario(scen_dir)
        p = best_practices_yaml_path(scen_dir)
        result = write_yaml_file_safely(p, content)
        if result.ok and sess is not None:
            sess.record_scenario_edit(target="best_practices", payload={"chars": len(content or "")})
        ctx = _scenario_editor_ctx(scen_dir, slug=slug, section="best_practices")
        raw = content
        if result.ok:
            raw = read_yaml_text(p)
        ctx.update(
            {
                "best_practices_raw": raw,
                "best_practices_rows": _yaml_line_rows(raw, result.error_line),
                "error": result.error_message if not result.ok else "",
                "message": "Best-practices YAML saved." if result.ok else "",
                "error_line": result.error_line,
            }
        )
        resp = render("partials/scenario_editor/section_best_practices.html", request, **ctx)
        if not result.ok:
            resp.status_code = 422
        return resp

    @app.post("/scenarios/{slug}/copy")
    async def scenarios_copy(slug: str) -> RedirectResponse:
        src = _safe_scenario_dir(scenarios_root, slug)
        ts = int(time.time())
        dst_name = f"{slug}__copy-{ts}"
        dst = scenarios_root / dst_name
        shutil.copytree(src, dst)
        return RedirectResponse(url=f"/scenarios/{dst_name}", status_code=303)

    @app.post("/scenarios/{slug}/copy_and_edit")
    async def scenarios_copy_and_edit(slug: str) -> RedirectResponse:
        src = _safe_scenario_dir(scenarios_root, slug)
        ts = int(time.time())
        dst_name = f"{slug}__copy-{ts}"
        dst = scenarios_root / dst_name
        shutil.copytree(src, dst)
        return RedirectResponse(url=f"/scenarios/{dst_name}/edit", status_code=303)

    @app.post("/scenarios/{slug}/fork")
    async def scenarios_fork_alias(slug: str) -> RedirectResponse:
        return RedirectResponse(url=f"/scenarios/{slug}/copy", status_code=307)

    @app.get("/runs/{run_path:path}/at/{turn}", response_class=HTMLResponse)
    async def runner_at_turn(request: Request, run_path: str, turn: int) -> HTMLResponse:
        return _runner_page(request, run_path, turn=turn)

    @app.get("/runs/{run_path:path}", response_class=HTMLResponse)
    async def runner(request: Request, run_path: str) -> HTMLResponse:
        return _runner_page(request, run_path, turn=None)

    return app
