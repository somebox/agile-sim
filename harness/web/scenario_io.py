"""Scenario editor IO helpers and lock-state lookup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import re
from typing import Any

import yaml

from harness.web.run_session import RunSession


@dataclass
class ScenarioLockState:
    run_in_progress: bool
    run_id: str | None = None
    run_url_path: str | None = None


@dataclass
class YamlWriteResult:
    ok: bool
    error_message: str = ""
    error_line: int | None = None
    error_column: int | None = None


def setting_path_for_scenario_dir(scenario_dir: Path) -> Path:
    """Resolve setting markdown path using `setting_file` from scenario.yaml."""

    raw = yaml.safe_load((scenario_dir / "scenario.yaml").read_text(encoding="utf-8")) or {}
    rel = str(raw.get("setting_file") or "setting.md").strip() or "setting.md"
    return scenario_dir / rel


def load_scenario_yaml(scenario_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((scenario_dir / "scenario.yaml").read_text(encoding="utf-8")) or {}


def write_scenario_yaml(scenario_dir: Path, data: dict[str, Any]) -> Path:
    p = scenario_dir / "scenario.yaml"
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(p.parent), delete=False) as tf:
        tf.write(text)
        tmp = Path(tf.name)
    tmp.replace(p)
    return p


def read_setting_md(scenario_dir: Path) -> str:
    p = setting_path_for_scenario_dir(scenario_dir)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def write_setting_md(scenario_dir: Path, content: str) -> Path:
    p = setting_path_for_scenario_dir(scenario_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = content or ""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(p.parent), delete=False) as tf:
        tf.write(body)
        tmp = Path(tf.name)
    tmp.replace(p)
    return p


def character_md_path(scenario_dir: Path, character: dict[str, Any]) -> Path:
    cid = str(character.get("id") or "").strip()
    rel = str(character.get("markdown_file") or f"characters/{cid}.md").strip()
    return scenario_dir / rel


def is_shared_character_path(scenario_dir: Path, path: Path) -> bool:
    """True if a character's markdown file lives outside its scenario folder.

    Such files come from the shared library (`scenarios/_characters/`) and
    must not be edited or deleted through a single scenario's editor, since
    other scenarios may reference the same file.
    """

    try:
        path.resolve().relative_to(scenario_dir.resolve())
        return False
    except ValueError:
        return True


_CHAR_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def validate_character_id(char_id: str) -> str:
    cid = str(char_id or "").strip()
    if not _CHAR_ID_RE.match(cid):
        raise ValueError("character id must match ^[a-z][a-z0-9_-]{0,31}$")
    return cid


def upsert_character(
    scenario_dir: Path,
    *,
    char_id: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    data = load_scenario_yaml(scenario_dir)
    chars = list(data.get("characters") or [])
    cid = validate_character_id(char_id)
    idx = -1
    for i, row in enumerate(chars):
        if str(row.get("id", "")).strip() == cid:
            idx = i
            break
    if idx < 0:
        raise ValueError("character not found")
    row = dict(chars[idx])

    if "name" in fields:
        row["name"] = str(fields["name"] or "").strip()
    if "role" in fields:
        row["role"] = str(fields["role"] or "").strip()
    if "sprite_set" in fields:
        row["sprite_set"] = str(fields["sprite_set"] or "").strip()
    if "model" in fields:
        row["model"] = str(fields["model"] or "").strip()
    iv = dict(row.get("initial_vitals") or {})
    for k in ("energy", "motivation", "stress"):
        if k in fields and fields[k] is not None and str(fields[k]).strip() != "":
            try:
                v = int(fields[k])
            except (TypeError, ValueError) as err:
                raise ValueError(f"{k} must be an integer") from err
            if v < 0 or v > 100:
                raise ValueError(f"{k} must be between 0 and 100")
            iv[k] = v
    row["initial_vitals"] = iv
    chars[idx] = row
    data["characters"] = chars
    write_scenario_yaml(scenario_dir, data)
    return row


def write_character_backstory(
    scenario_dir: Path,
    *,
    char_id: str,
    content: str,
) -> Path:
    data = load_scenario_yaml(scenario_dir)
    chars = list(data.get("characters") or [])
    cid = validate_character_id(char_id)
    row = next((dict(c) for c in chars if str(c.get("id", "")).strip() == cid), None)
    if row is None:
        raise ValueError("character not found")
    p = character_md_path(scenario_dir, row)
    if is_shared_character_path(scenario_dir, p):
        raise ValueError(
            "this character's backstory is shared across scenarios "
            f"({p}); edit it in the shared library directly"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    body = content or ""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(p.parent), delete=False) as tf:
        tf.write(body)
        tmp = Path(tf.name)
    tmp.replace(p)
    return p


def create_character(scenario_dir: Path, *, char_id: str, name: str) -> dict[str, Any]:
    data = load_scenario_yaml(scenario_dir)
    chars = list(data.get("characters") or [])
    cid = validate_character_id(char_id)
    if any(str(c.get("id", "")).strip() == cid for c in chars):
        raise ValueError("character id already exists")
    row = {
        "id": cid,
        "name": str(name or cid).strip() or cid,
        "role": "",
        "sprite_set": f"char_{cid}",
        "markdown_file": f"characters/{cid}.md",
        "initial_vitals": {"energy": 50, "motivation": 50, "stress": 50},
    }
    chars.append(row)
    data["characters"] = chars
    write_scenario_yaml(scenario_dir, data)
    p = scenario_dir / row["markdown_file"]
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return row


def delete_character(scenario_dir: Path, *, char_id: str) -> None:
    data = load_scenario_yaml(scenario_dir)
    chars = list(data.get("characters") or [])
    cid = validate_character_id(char_id)
    row = next((dict(c) for c in chars if str(c.get("id", "")).strip() == cid), None)
    if row is None:
        raise ValueError("character not found")
    for wi in data.get("work_items") or []:
        if str(wi.get("owner_id", "")).strip() == cid:
            raise RuntimeError(f"character owns work item {wi.get('id')}")
    for ch in data.get("channels") or []:
        if cid in [str(x) for x in (ch.get("member_ids") or [])]:
            raise RuntimeError(f"character is a member of channel {ch.get('name')}")
    for tm in data.get("teams") or []:
        if cid in [str(x) for x in (tm.get("member_ids") or [])]:
            raise RuntimeError(f"character is a member of team {tm.get('id')}")
    data["characters"] = [c for c in chars if str(c.get("id", "")).strip() != cid]
    write_scenario_yaml(scenario_dir, data)
    p = character_md_path(scenario_dir, row)
    if p.exists() and not is_shared_character_path(scenario_dir, p):
        p.unlink()


def _character_ids(data: dict[str, Any]) -> set[str]:
    return {
        str(c.get("id", "")).strip()
        for c in (data.get("characters") or [])
        if str(c.get("id", "")).strip()
    }


def upsert_channel(
    scenario_dir: Path,
    *,
    name: str,
    channel_type: str,
    coach_engagement: str,
    member_ids: list[str],
) -> dict[str, Any]:
    data = load_scenario_yaml(scenario_dir)
    nm = str(name or "").strip()
    if not nm.startswith("#"):
        raise ValueError("channel name must start with #")
    ce = str(coach_engagement or "").strip().lower()
    if ce not in {"post", "read", "none"}:
        raise ValueError("coach_engagement must be post, read, or none")
    ctype = str(channel_type or "open").strip().lower()
    if ctype not in {"open", "group", "event", "dm"}:
        raise ValueError("channel type must be open, group, event, or dm")
    ids = _character_ids(data)
    mids = [str(x).strip() for x in (member_ids or []) if str(x).strip()]
    unknown = [x for x in mids if x not in ids]
    if unknown:
        raise ValueError(f"unknown channel members: {', '.join(unknown)}")
    channels = list(data.get("channels") or [])
    row = {"name": nm, "type": ctype, "coach_engagement": ce, "member_ids": mids}
    idx = next((i for i, ch in enumerate(channels) if str(ch.get("name", "")).strip() == nm), -1)
    if idx >= 0:
        prev = dict(channels[idx])
        prev.update(row)
        row = prev
        channels[idx] = row
    else:
        channels.append(row)
    data["channels"] = channels
    write_scenario_yaml(scenario_dir, data)
    return row


def delete_channel(scenario_dir: Path, *, name: str) -> None:
    data = load_scenario_yaml(scenario_dir)
    nm = str(name or "").strip()
    channels = list(data.get("channels") or [])
    data["channels"] = [ch for ch in channels if str(ch.get("name", "")).strip() != nm]
    write_scenario_yaml(scenario_dir, data)


def upsert_team(scenario_dir: Path, *, team_id: str, name: str, member_ids: list[str]) -> dict[str, Any]:
    data = load_scenario_yaml(scenario_dir)
    tid = str(team_id or "").strip()
    if not tid:
        raise ValueError("team id is required")
    ids = _character_ids(data)
    mids = [str(x).strip() for x in (member_ids or []) if str(x).strip()]
    unknown = [x for x in mids if x not in ids]
    if unknown:
        raise ValueError(f"unknown team members: {', '.join(unknown)}")
    teams = list(data.get("teams") or [])
    row = {"id": tid, "name": str(name or tid).strip() or tid, "member_ids": mids}
    idx = next((i for i, tm in enumerate(teams) if str(tm.get("id", "")).strip() == tid), -1)
    if idx >= 0:
        prev = dict(teams[idx])
        prev.update(row)
        row = prev
        teams[idx] = row
    else:
        teams.append(row)
    data["teams"] = teams
    write_scenario_yaml(scenario_dir, data)
    return row


def delete_team(scenario_dir: Path, *, team_id: str) -> None:
    data = load_scenario_yaml(scenario_dir)
    tid = str(team_id or "").strip()
    teams = list(data.get("teams") or [])
    data["teams"] = [tm for tm in teams if str(tm.get("id", "")).strip() != tid]
    write_scenario_yaml(scenario_dir, data)


def upsert_work_item(
    scenario_dir: Path,
    *,
    work_id: str,
    title: str,
    state: str,
    owner_id: str,
) -> dict[str, Any]:
    data = load_scenario_yaml(scenario_dir)
    wid = str(work_id or "").strip()
    if not wid:
        raise ValueError("work item id is required")
    ids = _character_ids(data)
    oid = str(owner_id or "").strip()
    if oid and oid not in ids:
        raise ValueError(f"unknown owner_id: {oid}")
    st = str(state or "backlog").strip().lower()
    if st not in {"backlog", "doing", "done", "parked"}:
        raise ValueError("work item state must be backlog, doing, done, or parked")
    items = list(data.get("work_items") or [])
    row = {"id": wid, "title": str(title or wid).strip() or wid, "state": st, "owner_id": oid}
    idx = next((i for i, wi in enumerate(items) if str(wi.get("id", "")).strip() == wid), -1)
    if idx >= 0:
        prev = dict(items[idx])
        prev.update(row)
        row = prev
        items[idx] = row
    else:
        items.append(row)
    data["work_items"] = items
    write_scenario_yaml(scenario_dir, data)
    return row


def delete_work_item(scenario_dir: Path, *, work_id: str) -> None:
    data = load_scenario_yaml(scenario_dir)
    wid = str(work_id or "").strip()
    items = list(data.get("work_items") or [])
    data["work_items"] = [wi for wi in items if str(wi.get("id", "")).strip() != wid]
    write_scenario_yaml(scenario_dir, data)


def update_goals(
    scenario_dir: Path,
    *,
    max_turns: str,
    max_stress_any: str,
    abort_stress_any: str,
    min_done_work_items: str,
    per_team_min_done: str,
    require_done_ids: list[str],
) -> dict[str, Any]:
    data = load_scenario_yaml(scenario_dir)
    goals = dict(data.get("goals") or {})
    ints = {
        "max_turns": max_turns,
        "max_stress_any": max_stress_any,
        "abort_stress_any": abort_stress_any,
        "min_done_work_items": min_done_work_items,
    }
    for k, raw in ints.items():
        if str(raw).strip() == "":
            continue
        try:
            goals[k] = int(raw)
        except ValueError as err:
            raise ValueError(f"{k} must be an integer") from err
    if str(per_team_min_done).strip() == "":
        goals.pop("per_team_min_done", None)
    else:
        try:
            goals["per_team_min_done"] = int(per_team_min_done)
        except ValueError as err:
            raise ValueError("per_team_min_done must be an integer") from err

    req = [str(x).strip() for x in (require_done_ids or []) if str(x).strip()]
    known = {str(w.get("id", "")).strip() for w in (data.get("work_items") or []) if w.get("id")}
    missing = [x for x in req if x not in known]
    if missing:
        raise ValueError(f"unknown require_done_ids: {', '.join(missing)}")
    goals["require_done_ids"] = req
    data["goals"] = goals
    write_scenario_yaml(scenario_dir, data)
    return goals


def _parse_param_value(raw: str) -> Any:
    text = str(raw)
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def update_parameters(scenario_dir: Path, *, keys: list[str], values: list[str]) -> dict[str, Any]:
    data = load_scenario_yaml(scenario_dir)
    out: dict[str, Any] = {}
    for idx, k in enumerate(keys):
        key = str(k).strip()
        if not key:
            continue
        val = values[idx] if idx < len(values) else ""
        out[key] = _parse_param_value(str(val))
    data["parameters"] = out
    write_scenario_yaml(scenario_dir, data)
    return out


def process_yaml_path(scenario_dir: Path) -> Path:
    return scenario_dir / "process.yaml"


def best_practices_yaml_path(scenario_dir: Path) -> Path:
    data = load_scenario_yaml(scenario_dir)
    rel = str(data.get("best_practices_file") or "best_practices.yaml").strip() or "best_practices.yaml"
    return scenario_dir / rel


def read_yaml_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_yaml_file_safely(path: Path, content: str) -> YamlWriteResult:
    body = content or ""
    try:
        yaml.safe_load(body or "{}")
    except yaml.YAMLError as err:
        line = None
        col = None
        pm = getattr(err, "problem_mark", None)
        if pm is not None:
            line = int(getattr(pm, "line", 0)) + 1
            col = int(getattr(pm, "column", 0)) + 1
        if line is None:
            line = max(1, len(body.splitlines()))
        return YamlWriteResult(
            ok=False,
            error_message=str(err),
            error_line=line,
            error_column=col,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tf:
        tf.write(body)
        tmp = Path(tf.name)
    tmp.replace(path)
    return YamlWriteResult(ok=True)


def lock_state(
    scenario_dir: Path,
    sessions: dict[str, RunSession],
) -> ScenarioLockState:
    want = scenario_dir.resolve()
    for url_path, sess in sessions.items():
        if sess.bundle.path.resolve() != want:
            continue
        if sess.finished:
            continue
        return ScenarioLockState(run_in_progress=True, run_id=sess.run_id, run_url_path=url_path)
    return ScenarioLockState(run_in_progress=False, run_id=None, run_url_path=None)
