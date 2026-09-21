# Architecture (early sketch)

This is a first-pass shape, not a commitment. The goal here is to make the moving parts visible so we can argue about them.

## High-level shape

```
            ┌────────────────────────┐
            │       Web UI           │
            │  (scenario + runner)   │
            └───────────┬────────────┘
                        │  HTTP / WebSocket
            ┌───────────▼────────────┐
            │     API / Orchestrator │
            │  - scenario CRUD       │
            │  - run lifecycle       │
            │  - turn scheduler      │
            │  - timeline / events   │
            │  - cost/usage tracker  │
            └───────────┬────────────┘
                        │
        ┌───────────────┼─────────────────┬──────────────────┐
        │               │                 │                  │
   ┌────▼────┐    ┌─────▼──────┐    ┌─────▼──────┐    ┌──────▼─────┐
   │ Agent   │    │  Process   │    │   World    │    │   Comms    │
   │ layer   │◀──▶│  engine    │◀──▶│   ledger   │◀──▶│  (channels │
   │ (chars) │    │ (rules,    │    │  (data:    │    │  + msgs)   │
   │         │    │  validate, │    │  work,     │    │            │
   │         │    │  schedule) │    │  vitals,   │    │            │
   │         │    │            │    │  rels…)    │    │            │
   └────┬────┘    └────────────┘    └────────────┘    └────────────┘
        │
   ┌────▼─────┐
   │OpenRouter│
   │text+vis. │
   └──────────┘
```

The four core domain modules — agents, process engine, world ledger, comms — are peers. The orchestrator drives them through the turn lifecycle.

## Core components

### Web UI

**Implemented (read-only):** `harness/web` serves finished runs and batch folders via FastAPI — run picker, mission control (`/runs/…`), experiments (`/experiments/…`). CLI: `agile-harness serve`, `agile-harness view runs/<id>`. CSS build: `./scripts/build_css.sh`. Conventions: [`web-conventions.md`](web-conventions.md).

Planned on top of that:

- Scenario module: browse / edit / fork.
- Runner module: mission-control view, timeline, channels, vitals dashboard, coaching composer.
- Replay viewer.
- Server-rendered HTML over HTTP, with HTMX for partial updates and SSE for live turn progress / streaming agent output.

Stack: **HTMX + Alpine.js + Tailwind CSS**, served from FastAPI via Jinja templates. No npm or build pipeline; the Tailwind standalone CLI is the only build step. See [`ui-design.md`](ui-design.md) for the full UI design and rationale.

### API / Orchestrator (backend)

Owns the lifecycle of scenarios and runs. Responsibilities:

- **Scenario service** — CRUD on scenario definitions.
- **Run service** — create a run from a scenario, persist run state, expose snapshots.
- **Turn scheduler** — drives the turn lifecycle (see below): ticks the engine, builds per-agent context, dispatches agents, applies their outputs, recomputes metrics, persists.
- **Timeline / event store** — every meaningful thing that happens in a run is appended as a `TimelineEvent`. Replay is just playing the timeline back.
- **Cost/usage tracker** — token + spend accounting per run.

There is no separate "coaching service" — coach actions go through the same APIs agents use (`post`, `invoke`, `edit`, etc.) with `author = coach`, and per-channel engagement enforcement done by comms. The **coach actor** may be a human user (between-turn UI), an autonomous LLM (batch / autonomous mode), or a scripted preset; only the backing logic changes, not the API surface.

### Agent layer

One **agent runtime** instance per character in a run. Every agent is a character — a simulated person. Teams and the company are not agents; they are structural entries on the world ledger.

Each runtime:

- Holds the agent's profile (identity, personality, skills, goals).
- Maintains memory (recent turns, salient older events, summarized history).
- Builds the per-turn prompt from world ledger reads, own vitals, delivered messages, any coach-authored input from the previous turn, and the list of currently-active event channels the agent is a member of.
- Calls OpenRouter to produce: a narrative of the turn, statements/actions, async channel posts, process invocations, bounded self-reports on its own vitals, and — for each active event channel the agent is in — a **conversation contribution** (intent, current state, desired outcome).
- Decides how to interpret coaching — agreement, partial uptake, pushback, or ignoring it — based on its profile.

There is no separate "world resolver". Process-mediated actions (anything that touches the ledger structurally) are validated by the engine. Narrative outputs from different agents are simply recorded as-is — disagreements between agents are a feature, not something to reconcile.

### Process engine

Owns the **rules and the active behavior** on top of the world ledger. Deterministic; no LLM.

State it owns (rules + transient behavior state, *not* the data itself):

- **Rules** — roles, decision rights, approval gates, mandatory reviews, ritual cadence, working agreements.
- **Pending state** — open approvals, scheduled rituals queue, deadline calendar.

Interfaces:

- `consult(query) → answer` — read-only structured queries from any agent.
- `invoke(action) → result` — agent or coach-initiated process actions (escalate, request approval, introduce work, reassign, change deadline, schedule/cancel ritual, request channel, assert event). Validated against rules; executes (writing to ledger via the controlled APIs), queues, or rejects with a reason.
- `tick(turn)` — called by the orchestrator at the start of each simulation phase. Fires due rituals, expires timed-out approvals, applies deterministic vital rules, opens/archives event channels, surfaces deadline warnings.
- `edit(rule_change)` — used when the coach changes rules between turns. Versioned.

**Process invocation vocabulary (v1, harness).** Agents and the autonomous coach emit structured `process_invocations` in their turn output. Each item is an object with at least a string `kind` (and optional fields for args). The current Python hook in [`harness/engine.py`](../harness/engine.py) **records** recognized kinds on the run timeline as `process_invocation` events; it does **not** yet execute full policy (validate → queue → mutate ledger). Kinds in the v1 allowlist: `consult`, `invoke`, `tick`, `request_approval`, `change_deadline`, `edit_ritual`, `set_gate`. Any other non-empty `kind` is logged as `process_invocation_unhandled` for visibility. **Deferred:** richer kinds such as `introduce_work`, `escalate`, `reassign_person`, `open_channel`, and generic `schedule_ritual` / cancel — add when the engine applies rules and writes through to the ledger.

A scenario may also include a **process steward** character — an ordinary agent that has privileged access to the engine and operates it in-fiction (runs rituals, surfaces violations, talks to the coach). Optional; chaos scenarios omit it and run on a near-empty rulebook.

### World ledger

The structured data store. No rules, no LLM — just records, with controlled-write APIs (most writes go through the engine).

State it owns:

- **Work items** — id, title, owner(s), state, estimate, actual effort, dependencies, blocked-by.
- **Per-person turn record** — for each person each turn: capacity, planned commitments, delivered work. Replaces what were previously separate `Capacity` and `Commitment` entities.
- **Teams and org structure** — teams, members, reporting lines.
- **Relationships** — pair-wise notes between characters (worked together, conflicts, favors owed).
- **Vitals and metrics** — numeric values per character, per team, per org. Definitions come from the standard library + scenario extensions; per-turn snapshots are kept for history.
- **Engine event history** — every consult, invoke, rule edit, scheduled fire.

Read by everyone. Written by the engine (for process actions), by comms (for messages), and by the orchestrator (for vital deltas after clamping). Agents never write the ledger directly.

### Comms subsystem

Owns channels, identities, the message store, and the **narrator** that renders event-channel conversations.

The narrator is the only LLM-using piece of comms. Everything else is deterministic.

State:

- **Channels** — id, name, type (`dm` / `group` / `open` / `event`), members, lifecycle status, coach engagement mode (`post` / `read` / `none`), created-on-turn, archived-on-turn, optional bound-event id.

**DM channels (`dm/<character_id>`).** A DM is a **first-class** channel type, not a special case in the UI only. Naming is stable: one private thread between the coach and a given character (that character's id in the path). Membership is exactly `{ coach, that character }`; scenarios do not duplicate the same relationship as an extra row under `channels[]` in YAML — see [`simulation-model.md`](simulation-model.md). Coach engagement on DM is typically `post` for both sides; agent policy may restrict which DMs an agent may use (e.g. own DM only).
- **Identities** — display name + handle per character. Defined by the scenario.
- **Messages** — id, channel id, author, turn posted, content, parent message id, parsed mention ids.
- **Delivery state** — per-agent high-water mark + a separate mention queue (mentions get guaranteed delivery).

Interfaces:

- `post(channel, author, content, parent?)` — append a message; parse `@handle` mentions; update mention queues. Used for async posts in persistent channels.
- `read(agent, since_turn)` — return unread messages from member channels + any unread mentions.
- `open_channel(...)` / `archive_channel(...)` — driven by the engine for event channels.
- `request_channel(...)` — proxied through engine `invoke`; engine validates against scenario rules before calling `open_channel`.
- `narrate_event(channel, contributions, ledger_context) → (transcript, structured_outcome)` — for each active event channel, called once per turn after agents have run. Takes participants' contributions plus relevant ledger state; returns an ordered list of messages (each authored by a specific participant) and a structured outcome record (decisions reached, commitments made, unresolved conflicts). Transcript messages are appended via `post`; the structured outcome is handed to the engine to apply to the ledger.

Two activity modes:

- **Async posts** in persistent channels (DM / group / open). Posted on turn N, delivered on turn N+1.
- **Narrated conversations** in event channels. Members contribute intent/desired outcome in their normal turn output; the narrator renders the transcript in one call. Visible to participants the same turn; non-participants see the messages on turn N+1 like any other.

Channels are scenario-defined. Agents do not freely create channels; they request via `invoke`.

The coach is a participant subject to per-channel engagement mode. Coach posts go through `post` like any other author. If the coach is a member of an active event channel, her latest input becomes a contribution to the next narration.

### Boundary: orchestrator / engine / ledger / comms

To keep concerns from blurring:

- **Orchestrator** — domain-agnostic. Drives the turn lifecycle. Doesn't know what "approval" means.
- **Process engine** — owns *rules and validation*. Anything that requires policy goes through it.
- **World ledger** — owns *data*. Holds the answer to "what is true right now". No behavior beyond CRUD and aggregation.
- **Comms** — owns *channels and messages*. Adjacent to the ledger but kept separate because it has its own access patterns (delivery state, mentions).

Mechanical = orchestrator. Policy = engine. Data = ledger. Communication = comms.

### Storage

File-first. Keep dependencies and operational complexity low; reach for a database only when files start to hurt.

```
<deploy-root>/
├── global/                        # global, deploy-wide config
│   ├── best_practices.yaml        # coaching best-practices library
│   ├── sprite_library/            # standard sprite sets
│   ├── vital_defaults.yaml        # standard library of vitals/metrics
│   └── settings.yaml              # global app settings (model tiers, etc.)
│
├── scenarios/
│   ├── _characters/                # shared character library, reused across scenarios
│   │   ├── priya.md
│   │   └── marcus-chen.md
│   ├── two-teams-one-integration/
│   │   ├── scenario.yaml          # structured definition
│   │   ├── setting.md             # free-text world description
│   │   ├── characters/            # scenario-local characters (not reused elsewhere)
│   │   │   └── lin.md
│   │   ├── process.yaml           # rules / rituals / decision rights
│   │   ├── best_practices.yaml    # scenario-specific additions
│   │   └── assets/                # scenario-bound images, sprite overrides
│   └── …
│
└── runs/
    └── <run-id>/
        ├── run.yaml               # metadata, scenario ref + version
        ├── timeline.jsonl         # append-only TimelineEvent stream
        ├── snapshots/             # periodic ledger snapshots
        ├── summary.json           # end-of-run summary
        ├── final_report.md        # generated long-form report
        └── assets/                # generated images for this run
```

A scenario is a folder of YAML + markdown — versionable in git, hand-editable, diff-friendly. The structured fields live in YAML; prose (settings, character backstories) lives in markdown files referenced from the YAML. This keeps prose in a comfortable editor and structure in a parseable form.

**Reusing characters across scenarios.** A character's `markdown_file` is just a path relative to the scenario folder, so it can point outside it — e.g. `markdown_file: ../_characters/priya.md` to reuse a persona from the shared library at `scenarios/_characters/`. A scenario's own `characters[]` entry still defines that character's id, name, role, sprite, and `initial_vitals` locally (those can differ per scenario even when the backstory is shared); only the prose backstory is shared. The scenario editor treats any character whose markdown file resolves outside its own folder as read-only — edit the shared file directly, or copy it into the scenario's own `characters/` folder first if you want to fork the backstory.

The global folder lives wherever the app is deployed and holds the standard library: best-practices library, vital defaults, sprites, app settings. Scenarios may override or extend.

A run is its own folder under `runs/`. The timeline is an append-only `.jsonl` file — that's the source of truth; everything else is derived from replaying it. Periodic snapshots avoid replaying from turn 0 every time the UI loads a run.

**SQLite** is allowed if it earns its place — likely candidates are run-list browsing, full-text message search, or aggregating usage records across runs. v1 starts file-only; an index DB gets added when file-walking gets slow, not pre-emptively.

### OpenRouter integration

A thin client wrapping:

- Text completions for agent reasoning and chat.
- Vision/image generation for avatars and scenes.
- Per-call metadata (model, tokens, cost) flowing back to the usage tracker.

Model selection is configurable per agent role.

## Turn lifecycle

A run alternates between a **simulation phase** (agents act) and a **coaching phase** (user acts). The coaching phase is half the loop, not a side channel.

### Simulation phase

1. **Tick the engine** — fire due rituals, expire timed-out approvals, apply deterministic vital rules (e.g. overload → stress), open/archive event channels, surface deadline warnings.
2. **Build per-agent context** — for each agent: relevant ledger reads, own vitals, unread messages from comms (including mentions), any coach-authored input from the prior coaching phase, and the list of active event channels they are a member of.
3. **Run agents** — in parallel where independent. Each agent's single LLM call produces: narrative, async channel posts, process invocations, vital self-report deltas, and one conversation contribution per active event channel they belong to.
4. **Apply structural outputs** — engine validates and applies invocations (writing to the ledger); async messages stored in comms; vital deltas clamped and applied.
5. **Narrate event channels** — for each currently-active event channel, the comms narrator renders the conversation from members' contributions and relevant ledger context, producing (a) an ordered transcript posted to the channel and (b) a structured outcome record. The engine applies the outcome to the ledger (commitments, decisions, work-item updates, vital nudges if warranted). One LLM call per active event channel.
6. **Recompute aggregates** — org-level metrics derived from the ledger.
7. **Check exit conditions** — including thresholds expressed in vitals/metrics. If hit, mark the run done.
8. **Persist & notify** — append all events to the timeline; notify UI; enter the coaching phase.

### Coaching phase

9. UI surfaces the new state, channel activity (including any narrated transcripts), vital changes, and engine events. The coach inspects.
10. The coach takes zero or more actions, each appended to the timeline as it happens:
   - Post in any channel (DM / group / open) where engagement mode allows.
   - `invoke` a process action (event injection is just an `invoke` with the coach as author).
   - `edit` engine rules.
   - Change scenario parameters.
   - Nudge a vital directly (recorded as source `coaching`).
   - Skip ("do nothing, advance").

When the coach advances the turn, control returns to step 1.

## Data model (sketch)

- `Scenario` — id, name, setting, process definition, character definitions, entity definitions (teams, org), initial state, parameters, goal, exit conditions, turn duration unit.
- `Run` — id, scenario id and version, status, started/ended timestamps, current turn, live parameters, totals (tokens/cost).
- `Character` — run-scoped instance: profile (personality, skills, goals), memory, current internal view. (Replaces what was called `Agent`.)
- `Identity` — character id, display name, handle, sprite set reference.
- `SpriteSet` — id, image references for each expression state (`idle`, `happy`, `frustrated`, `overloaded`, `bored`, `surprised`, `proud`, `sad`). From the standard library or generated per scenario.
- `Team` — id, name, member character ids.
- `WorkItem` — id, title, owner(s), state, estimate, actual effort, dependencies, blocked-by.
- `PersonTurnRecord` — per character per turn: capacity, committed items, delivered items, adjustments.
- `Relationship` — pair-wise notes between characters.
- `VitalDefinition` — name, scope (`person` / `team` / `org`), range, default value, update rules, description. From the standard library or scenario.
- `BestPractice` — id, name, category, description, optional weight. Global library + scenario additions/overrides; consumed only by the Summary loop, not by simulation.
- `VitalValue` — entity ref, vital name, turn, value, source of last change.
- `ProcessDefinition` — rulebook portion of a scenario: roles, decision rights, gates, ritual cadence, work-item state machine.
- `ProcessState` — run-scoped: pending approvals, scheduled rituals queue, current rule version.
- `Channel` — id, name, type, member ids, lifecycle status, coach engagement mode, created/archived turns, optional bound-event id.
- `Message` — id, channel id, author (character id or `coach`), turn, content, parent message id, parsed mention ids, source (`post` for direct posts, `narrated` for narrator-rendered).
- `Reaction` — message id, reactor (character id or `coach`), emoji, turn.
- `EventConversation` — one record per narrated event-channel conversation in a turn: channel id, turn, participants, per-participant contributions (intent, state, desired outcome), structured outcome (decisions, commitments, unresolved conflicts), reference to the resulting `Message`s.
- `TimelineEvent` — the universal record of "something happened". Fields: run id, turn, sequence, author (character id / `coach` / `engine` / `system`), kind (e.g. `message`, `invoke`, `invoke_result`, `rule_edit`, `parameter_change`, `vital_update`, `engine_tick`, `agent_narrative`, `exit_condition_fired`), payload. Every other entity above either *is* something a `TimelineEvent` references, or is the current snapshot derived from replaying timeline events.
- `Asset` — id, kind (avatar / background / scene), prompt, generated image ref.
- `UsageRecord` — call-level metadata (model, tokens, cost, latency).

The previous `CoachingInput` entity is gone — coach actions are just `TimelineEvent`s with `author = coach`.

## Scenario definition

A scenario is a **folder** (see "Storage" above) combining structured YAML and free-text markdown:

- `scenario.yaml` — typed fields: process rules, exit conditions, parameter ranges, channel definitions, vital extensions, character templates, references to markdown files.
- `setting.md` — free-text world description, used directly in prompts.
- `characters/<name>.md` — per-character backstory and personality, referenced from `scenario.yaml`.
- `process.yaml` — rulebook (decision rights, gates, ritual cadence).
- `best_practices.yaml` — scenario-specific additions to the global library.

The scenario module reads/writes this folder. Structure stays parseable; prose stays in a markdown editor; both diff cleanly in git.

## Cross-cutting concerns

- **Determinism** — _(open)_ runs may be seeded for reproducibility. Even with seeds, model nondeterminism limits this.
- **Cost control** — per-turn token budgets, per-agent model tiers, max-turn caps.
- **Memory growth** — periodic summarization of older turns into condensed agent memory.
- **Failure handling** — model call failures should fail the turn cleanly, leaving prior state intact and allowing retry.
- **Observability** — structured logs of every model call, prompt, and response (these *are* the data of a run).

## Tech choices

- **Backend**: Python + FastAPI. (Confirmed: PydanticAI in the agent layer is Python-only; HTMX server-rendering pairs naturally with FastAPI/Jinja.)
- **Frontend**: HTMX + Alpine.js + Tailwind CSS, served as Jinja templates from the backend. No SPA, no npm. See [`ui-design.md`](ui-design.md).
- **Storage**: filesystem-first (YAML + markdown for scenarios, JSONL timelines for runs); SQLite added only if a workload demands an index. See "Storage" above.
- **Realtime**: SSE (preferred) or WebSocket for turn progress and chat streaming, integrated via `hx-ext="sse"`.
- **Agent runtime**: PydanticAI agents calling OpenRouter; Langfuse for observability. See [`agentic-design.md`](agentic-design.md).
- **Packaging**: single repo. `uv`/`pip` for Python deps; Tailwind standalone CLI binary committed or downloaded at install. Run locally via a simple `dev` script; Docker Compose only if needed.

## Build order

Grouped into three milestones so we have a clear "definition of done" for each.

### Foundation MVP — proves the loop

1. Scenario data model + minimal scenario CRUD (backend + UI list/edit).
2. Run lifecycle with a stub agent (deterministic, no model calls). Timeline event store. Proves the simulation/coaching loop end-to-end.
3. Minimal world ledger: work items, per-person turn records, teams.
4. Minimal process engine: a few rules, `consult` and `invoke` against the ledger, `tick`. No steward yet.
5. Comms subsystem: channels, messages, async delivery, `@handle` mentions. Coach can DM the stub agent and the message lands next turn.

### First useful run — real agents and real coaching

6. OpenRouter text integration; one real character agent that consults/invokes the engine and posts to channels.
7. Vitals on the ledger: standard library defaults, deterministic rules, agent self-reports clamped, coach nudges. Vitals dashboard in UI.
8. Process steward character + coach-facing rule editor.
9. Multi-character turns. Cross-character channel threads start to emerge.
10. Coach actions: parameter changes, event injections (as engine `invoke`s), rule edits.

### Polish

11. Replay viewer (replays the timeline event stream).
12. Vision-generated visuals (avatars first, then scenes/backgrounds).
13. Cost tracking + UI polish.
