"""Recurring in-session wakeups — the /loop command (Claude Code parity).

A loop stops when the agent ends a wakeup reply with ``LOOP_COMPLETE`` on its own line, when
``--times N`` ticks have fired, when the ``--until`` judge rules the condition met, or when the
``loops.max_ticks`` backstop pauses it. State lives in SessionDB ``state_meta`` (same contract as
``hermes_cli/goals.py``); CLI, gateway, and TUI all drive it through :class:`LoopManager`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Floor for fixed intervals. Claude Code allows 30s; anything tighter is almost always an
# accident that burns tokens polling unchanged state. Config loops.min_interval_seconds (clamped ≥ 5).
DEFAULT_MIN_INTERVAL_SECONDS = 30
# Backstop tick budget so an unattended loop can't run forever. 0 = unlimited; config loops.max_ticks.
DEFAULT_MAX_TICKS = 100
# Self-paced mode: start at the floor, double while replies are unchanged, cap at the
# ceiling, snap back to the floor on any change.
DEFAULT_SELF_PACED_FLOOR_SECONDS = 60
DEFAULT_SELF_PACED_CEILING_SECONDS = 15 * 60

# Completion sentinel the wakeup prompt teaches the agent to emit.
LOOP_COMPLETE_MARKER = "LOOP_COMPLETE"
# Marker on its own line, tolerating surrounding whitespace / trailing punctuation.
_LOOP_COMPLETE_RE = re.compile(
    r"(?im)^\s*" + re.escape(LOOP_COMPLETE_MARKER) + r"\s*[.!]?\s*$"
)
# Interval token: 30s / 5m / 2h / 1h30m (compound units allowed, at least one).
_INTERVAL_TOKEN_RE = re.compile(
    r"^(?=\d)(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.IGNORECASE
)


WAKEUP_PROMPT_TEMPLATE = (
    "[/loop wakeup #{tick}{cadence}]\n"
    "Recurring task: {prompt}\n\n"
    "This is an automatic wakeup from the /loop the user set. Perform the "
    "task now against the CURRENT state (re-check files, processes, or "
    "services fresh — do not assume anything from earlier iterations still "
    "holds). Report concisely what you found or did this iteration.\n"
    "If the task is now complete, no longer applicable, or the thing you "
    "were watching has finished, say so and end your reply with "
    f"{LOOP_COMPLETE_MARKER} on its own line — that stops the loop."
)

WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE = (
    "[/loop wakeup #{tick}{cadence}]\n"
    "Recurring task: {prompt}\n\n"
    "Stop condition: {until}\n\n"
    "This is an automatic wakeup from the /loop the user set. Perform the "
    "task now against the CURRENT state (re-check files, processes, or "
    "services fresh — do not assume anything from earlier iterations still "
    "holds). Report concisely what you found or did this iteration, and "
    "show concrete evidence of the stop condition's status.\n"
    "If the stop condition is met, or the task is no longer applicable, say "
    f"so and end your reply with {LOOP_COMPLETE_MARKER} on its own line — "
    "that stops the loop."
)


def parse_interval_token(token: str) -> Optional[int]:
    """Total seconds for ``30s``/``5m``/``2h``/``1h30m``, else None.

    A bare number is NOT an interval (it collides with prompt text like ``/loop 3 things``).
    """
    m = _INTERVAL_TOKEN_RE.match(token.strip()) if token else None
    if not m:
        return None
    h, mnt, s = (int(g) if g else 0 for g in m.groups())
    total = h * 3600 + mnt * 60 + s
    return total if total > 0 else None


def parse_loop_args(text: str) -> dict[str, Any]:
    """Parse ``/loop [interval] <prompt> [--times N] [--until ...]``.

    Returns ``{"interval_seconds": int|None, "prompt", "times", "until", "error"}``;
    ``interval_seconds`` None means self-paced, ``error`` is set for unusable input.
    """
    raw = (text or "").strip()
    result: dict[str, Any] = {"interval_seconds": None, "prompt": "", "times": 0, "until": "", "error": None}
    if not raw:
        return {**result, "error": "empty"}

    # Pull trailing flags first so an interval-looking token inside the --until clause can't
    # confuse the front parse. --until consumes to end-of-line (or to a following --times).
    times, until = 0, ""
    m_times = re.search(r"\s--times\s+(\S+)", raw)
    if m_times:
        try:
            times = int(m_times.group(1))
            if times < 1:
                raise ValueError
        except ValueError:
            return {**result, "error": f"--times expects a positive integer, got {m_times.group(1)!r}"}
        raw = (raw[: m_times.start()] + raw[m_times.end():]).strip()

    m_until = re.search(r"\s--until\s+(.+)$", raw, re.DOTALL)
    if m_until:
        until = m_until.group(1).strip()
        raw = raw[: m_until.start()].strip()

    # Leading "every" sugar: /loop every 5m <prompt>
    tokens = raw.split(None, 1)
    if tokens and tokens[0].lower() == "every" and len(tokens) > 1:
        raw = tokens[1]
        tokens = raw.split(None, 1)

    interval = parse_interval_token(tokens[0]) if tokens else None
    if interval is not None:
        raw = tokens[1].strip() if len(tokens) > 1 else ""

    if not raw:
        return {**result, "error": "missing prompt (usage: /loop [interval] <prompt>)"}
    return {**result, "interval_seconds": interval, "prompt": raw, "times": times, "until": until}


def format_interval(seconds: float) -> str:
    """Render seconds as a compact human interval (``90`` → ``1m30s``)."""
    h, rem = divmod(int(max(0, round(seconds))), 3600)
    m, s = divmod(rem, 60)
    parts = [f"{h}h"] if h else []
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return "".join(parts)


def _loops_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("loops") or {}
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _config_int(key: str, default: int, floor: int) -> int:
    """``loops.<key>`` as an int clamped to ``floor``; ``default`` on any bad value."""
    try:
        return max(floor, int(_loops_config().get(key, default)))
    except Exception:
        return default


def min_interval_seconds() -> int:
    return _config_int("min_interval_seconds", DEFAULT_MIN_INTERVAL_SECONDS, 5)


def max_ticks_default() -> int:
    return _config_int("max_ticks", DEFAULT_MAX_TICKS, 0)


def self_paced_floor_seconds() -> int:
    return _config_int("self_paced_floor_seconds", DEFAULT_SELF_PACED_FLOOR_SECONDS, 10)


def self_paced_ceiling_seconds() -> int:
    floor = self_paced_floor_seconds()
    return _config_int("self_paced_ceiling_seconds", max(floor, DEFAULT_SELF_PACED_CEILING_SECONDS), floor)


@dataclass
class LoopState:
    """Serializable /loop state stored per session."""

    prompt: str
    status: str = "active"            # active | paused | done | cleared
    mode: str = "interval"            # interval | self_paced | hybrid
    interval_seconds: float = 0.0     # fixed cadence (mode == "interval")
    current_delay: float = 0.0        # live cadence (self-paced backoff)
    times: int = 0                    # user cap (--times N); 0 = none
    until: str = ""                   # judged stop condition; "" = none
    max_ticks: int = DEFAULT_MAX_TICKS  # config backstop; 0 = unlimited
    ticks_fired: int = 0
    created_at: float = 0.0
    last_fired_at: float = 0.0
    next_due_at: float = 0.0
    # True between "wakeup injected" and "that turn's response evaluated": stops a tick from
    # double-firing mid-turn and tells the post-turn hook the turn that just ended was ours.
    awaiting_response: bool = False
    last_response_digest: str = ""    # self-paced change detection
    paused_reason: Optional[str] = None
    last_stop_reason: Optional[str] = None
    # Gateway routing (platform / chat_id / chat_type / thread_id) captured at creation so the
    # idle watcher can inject ticks into the right chat. Empty for CLI/TUI (own schedulers).
    route: dict[str, str] = field(default_factory=dict)
    # TaskController hybrid mode: persisted binding and read-only observer high-water mark.
    taskcontroller_binding: dict[str, Any] = field(default_factory=dict)
    observer_cursor: dict[str, Any] = field(default_factory=dict)
    observer_interval_seconds: float = 0.0
    observer_checks: int = 0
    processed_event_keys: list[str] = field(default_factory=list)
    pending_events: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> LoopState:
        data = json.loads(raw)
        route = data.get("route")
        kwargs: dict[str, Any] = {
            "prompt": data.get("prompt", ""),
            "status": data.get("status", "active"),
            "mode": data.get("mode", "interval"),
            "paused_reason": data.get("paused_reason"),
            "last_stop_reason": data.get("last_stop_reason"),
            "route": route if isinstance(route, dict) else {},
        }
        # Collections need explicit defaults because their dataclass defaults are factories.
        collection_defaults = {
            "taskcontroller_binding": dict,
            "observer_cursor": dict,
            "processed_event_keys": list,
            "pending_events": list,
        }
        casts = {"str": str, "int": int, "float": float, "bool": bool}
        for f in fields(cls):
            if f.name in kwargs:
                continue
            if f.name in collection_defaults:
                value = data.get(f.name)
                factory = collection_defaults[f.name]
                kwargs[f.name] = value if isinstance(value, factory) else factory()
            else:
                kwargs[f.name] = casts[f.type](data.get(f.name, f.default) or casts[f.type]())
        return cls(**kwargs)

    def cadence_label(self) -> str:
        if self.mode == "self_paced":
            live = f", currently {format_interval(self.current_delay)}" if self.current_delay else ""
            return f"self-paced{live}"
        return f"every {format_interval(self.interval_seconds)}"

    def remaining_label(self) -> str:
        if self.status != "active":
            return ""
        remaining = self.next_due_at - time.time()
        return "due now" if remaining <= 0 else f"next in {format_interval(remaining)}"


_META_PREFIX = "loop:"


def _meta_key(session_id: str) -> str:
    return f"{_META_PREFIX}{session_id}"


def _get_session_db() -> Optional[Any]:
    """The goals module's cached SessionDB, so goals/loops/heartbeats share one connection and
    its off-loop bootstrap (a cold cache on the loop thread never runs ``SessionDB()`` inline).

    The previous copy here did, which froze the loop for the init duration and dropped the first ``loop:*``
    write (the /goal bug class, #88965).
    """
    try:
        from hermes_cli.goals import _get_session_db as _goals_db
    except Exception as exc:  # pragma: no cover
        logger.debug("LoopManager: SessionDB bootstrap failed (%s)", exc)
        return None
    return _goals_db()


def _db_op(label: str, fn, default=None):
    """Run one SessionDB call; any error is logged at debug and yields ``default``."""
    try:
        return fn()
    except Exception as exc:
        logger.debug("LoopManager: %s failed: %s", label, exc)
        return default


def _parse_state(raw: str, session_id: str = "") -> Optional[LoopState]:
    """``LoopState`` from stored JSON; None (warning when *session_id* given) on corrupt data."""
    try:
        return LoopState.from_json(raw)
    except Exception as exc:
        if session_id:
            logger.warning("LoopManager: could not parse stored loop for %s: %s", session_id, exc)
        return None


def load_loop(session_id: str) -> Optional[LoopState]:
    """Load the loop for a session, or None if none exists."""
    db = _get_session_db() if session_id else None
    if db is None:
        return None
    raw = _db_op("get_meta", lambda: db.get_meta(_meta_key(session_id)))
    return _parse_state(raw, session_id) if raw else None


def save_loop(session_id: str, state: LoopState) -> None:
    """Persist a loop to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        from hermes_cli.goals import _warn_dropped_write

        _warn_dropped_write("LoopManager", "loop", session_id)
        return
    _db_op("set_meta", lambda: db.set_meta(_meta_key(session_id), state.to_json()))


def clear_loop(session_id: str) -> None:
    """Mark a loop cleared in the DB (preserved for audit, status=cleared)."""
    state = load_loop(session_id)
    if state is not None:
        state.status = "cleared"
        save_loop(session_id, state)


def list_active_loops() -> list[tuple[str, LoopState]]:
    """``[(session_id, LoopState), ...]`` for every ACTIVE loop; ``[]`` on any DB error.

    Used by the gateway's idle wakeup watcher, which scans for due loops on a coarse tick.
    """
    db = _get_session_db()
    if db is None:
        return []
    out: list[tuple[str, LoopState]] = []
    for key, raw in _db_op("list_meta_prefix", lambda: db.list_meta_prefix(_META_PREFIX), []):
        session_id = key[len(_META_PREFIX):]
        state = _parse_state(raw) if session_id and raw else None
        if state is not None and state.status == "active":
            out.append((session_id, state))
    return out


def store_has_active_loop(db: Any) -> bool:
    """True when *db* holds an ACTIVE ``loop:*`` row — or a row that cannot be parsed (unknown, so the
    caller keeps its full scan). Unlike :func:`list_active_loops` this takes the store explicitly and
    propagates read errors, so an idle gate can tell "empty" from "unavailable"."""
    for key, raw in db.list_meta_prefix(_META_PREFIX):
        state = _parse_state(raw, key[len(_META_PREFIX):]) if raw else None
        if state is None or state.status == "active":
            return True
    return False


def migrate_loop_to_session(old_session_id: str, new_session_id: str, *, reason: str = "") -> bool:
    """Carry a /loop from a parent session to its continuation. Best-effort, never raises.

    Context compression rotates ``session_id`` to a fresh child; without this the loop silently
    dies at the compaction boundary.

    Copies the loop onto the new session and archives the old row as ``cleared`` so exactly one active loop
    row exists per logical conversation. See #33618.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_loop(old_session_id)
        if state is None or state.status == "cleared" or load_loop(new_session_id) is not None:
            return False
        save_loop(new_session_id, state)
        clear_loop(old_session_id)
        logger.debug(
            "LoopManager: migrated loop %s -> %s (%s)",
            old_session_id, new_session_id, reason or "rotation",
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("LoopManager: loop migration failed: %s", exc)
        return False


def _ticks_label(n: int) -> str:
    return f"{n} tick{'s' if n != 1 else ''}"


def _dash(reason: Optional[str]) -> str:
    return f" — {reason}" if reason else ""


def response_signals_complete(response: str) -> bool:
    """True when the agent ended its reply with the LOOP_COMPLETE marker."""
    return bool(response) and _LOOP_COMPLETE_RE.search(response) is not None


def _digest_response(response: str) -> str:
    """Digest for self-paced change detection; whitespace-normalized with clock/timestamp/duration
    tokens stripped so 'checked at 14:02:33' doesn't defeat the backoff."""
    text = (response or "").strip().lower()
    text = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


class LoopManager:
    """Per-session /loop state + tick decisions.

    Drivers call ``set``/``pause``/``resume``/``clear`` for user controls, ``is_due()`` (cheap,
    in-memory), ``fire_tick()`` to claim a tick and get the wakeup message, ``complete_tick()`` to
    evaluate the finished turn, and ``status_line()``.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._state: Optional[LoopState] = load_loop(session_id)

    @property
    def state(self) -> Optional[LoopState]:
        return self._state

    def refresh(self) -> None:
        """Re-read state from the DB (cross-process safety for the gateway)."""
        self._state = load_loop(self.session_id)

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_loop(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def _save(self) -> LoopState:
        save_loop(self.session_id, self._state)
        return self._state

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status == "cleared":
            return "No loop set. Start one with /loop [interval] <prompt>."
        fired = _ticks_label(s.ticks_fired)
        if s.times:
            caps = [f"{s.ticks_fired}/{s.times} runs"]
        elif s.max_ticks:
            caps = [f"{s.ticks_fired}/{s.max_ticks} budget"]
        else:
            caps = [fired]
        if s.until:
            caps.append(f"until: {s.until}")
        meta = f"{s.cadence_label()}, {', '.join(caps)}"
        if s.status == "active":
            remaining = s.remaining_label()
            tail = ", wakeup running" if s.awaiting_response else (f", {remaining}" if remaining else "")
            return f"↻ Loop (active, {meta}{tail}): {s.prompt}"
        if s.status == "paused":
            return f"⏸ Loop (paused, {meta}{_dash(s.paused_reason)}): {s.prompt}"
        if s.status == "done":
            return f"✓ Loop finished ({fired}{_dash(s.last_stop_reason)}): {s.prompt}"
        return f"Loop ({s.status}, {meta}): {s.prompt}"

    def set(
        self,
        prompt: str,
        *,
        interval_seconds: Optional[int] = None,
        times: int = 0,
        until: str = "",
        route: Optional[dict[str, str]] = None,
    ) -> LoopState:
        """Start a new loop (replaces any existing one for the session)."""
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("loop prompt is empty")

        now = time.time()
        self_paced = interval_seconds is None
        interval = 0.0 if self_paced else float(max(int(interval_seconds), min_interval_seconds()))
        state = LoopState(
            prompt=prompt,
            mode="self_paced" if self_paced else "interval",
            interval_seconds=interval,
            current_delay=float(self_paced_floor_seconds()) if self_paced else interval,
            times=max(0, int(times or 0)),
            until=(until or "").strip(),
            max_ticks=max_ticks_default(),
            created_at=now,
            next_due_at=now,
            route=dict(route or {}),
        )
        self._state = state
        return self._save()

    def set_hybrid(
        self,
        prompt: str,
        *,
        binding: dict[str, Any],
        observer_interval_seconds: int,
        observer_cursor: Optional[dict[str, Any]] = None,
    ) -> LoopState:
        """Bind this session to a TaskController event trigger plus bounded read-only observer.

        Timer due-ness only schedules an observer read; it never creates an LLM tick.
        The current canonical mailbox/continuation remain owned by TaskController.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("loop prompt is empty")
        if not isinstance(binding, dict):
            raise ValueError("TaskController binding must be an object")
        required = (
            "task_id", "run_id", "repository", "branch", "worktree", "head_sha",
            "session_id", "node_id", "executor_actor", "agent_instance", "capability",
            "observer_role", "controller_mailbox_ref", "executor_mailbox_ref", "mailbox_ref",
            "wakeup_binding",
        )
        if any(not isinstance(binding.get(key), str) or not binding[key].strip() for key in required):
            raise ValueError("TaskController binding is incomplete")
        if binding["session_id"] != self.session_id:
            raise ValueError("TaskController binding session mismatch")
        if binding["observer_role"] != "MAILBOX_OBSERVER":
            raise ValueError("TaskController Loop requires MAILBOX_OBSERVER")
        if binding["capability"] != "taskcontroller.executor":
            raise ValueError("TaskController executor capability mismatch")
        if binding["controller_mailbox_ref"] != binding["mailbox_ref"]:
            raise ValueError("TaskController controller mailbox binding mismatch")
        if re.fullmatch(r"[0-9a-f]{40}", binding["head_sha"]) is None:
            raise ValueError("TaskController head_sha must be a full lowercase commit SHA")
        if type(observer_interval_seconds) is not int or not (min_interval_seconds() <= observer_interval_seconds <= self_paced_ceiling_seconds()):
            raise ValueError("TaskController observer interval is outside the bounded range")
        cursor = observer_cursor if observer_cursor is not None else {
            "protocol": "dw.taskcontroller.mailbox-cursor/v1",
            "mailbox_ref": binding["controller_mailbox_ref"],
            "run_id": binding["run_id"],
            "node_id": binding["node_id"],
            "actor_namespace": "controller",
            "last_logical_seq": -1,
            "last_event_seq": -1,
            "last_event_id": None,
            "last_event_digest": None,
        }
        if not isinstance(cursor, dict) or any(
            cursor.get(key) != expected for key, expected in (
                ("mailbox_ref", binding["controller_mailbox_ref"]),
                ("run_id", binding["run_id"]),
                ("node_id", binding["node_id"]),
                ("actor_namespace", "controller"),
            )
        ):
            raise ValueError("TaskController observer cursor binding mismatch")
        now = time.time()
        self._state = LoopState(
            prompt=prompt,
            status="active",
            mode="hybrid",
            interval_seconds=float(observer_interval_seconds),
            current_delay=float(observer_interval_seconds),
            max_ticks=max_ticks_default(),
            created_at=now,
            next_due_at=now,
            taskcontroller_binding=dict(binding),
            observer_cursor=dict(cursor),
            observer_interval_seconds=float(observer_interval_seconds),
        )
        return self._save()

    def pause(self, reason: str = "user-paused") -> Optional[LoopState]:
        s = self._state
        if not s or s.status not in {"active", "paused"}:
            return None
        s.status, s.paused_reason, s.awaiting_response = "paused", reason, False
        return self._save()

    def resume(self) -> Optional[LoopState]:
        s = self._state
        if not s or s.status == "cleared":
            return None
        s.status, s.paused_reason, s.awaiting_response = "active", None, False
        # Re-arm relative to now so a long pause doesn't fire instantly N times.
        delay = s.current_delay or s.interval_seconds or self_paced_floor_seconds()
        s.next_due_at = time.time() + min(delay, 5.0)
        return self._save()

    def clear(self) -> bool:
        if self._state is None or self._state.status == "cleared":
            return False
        self._state.status = "cleared"
        self._save()
        self._state = None
        return True

    def is_due(self, now: Optional[float] = None) -> bool:
        """Cheap check: active, not mid-wakeup, and the clock has passed."""
        s = self._state
        return (
            s is not None and s.status == "active" and not s.awaiting_response
            and (now if now is not None else time.time()) >= s.next_due_at
        )

    def fire_tick(self) -> Optional[str]:
        """Claim a due tick; returns the message to inject, or None.

        The message is the wakeup-framed prompt, or the raw command when the loop's prompt is
        itself a slash command (``/loop 10m /recap``). Marks ``awaiting_response`` so the tick
        can't double-fire; drivers MUST follow up with ``complete_tick`` (or ``abandon_tick``).
        """
        s = self._state
        if s is None or not self.is_due():
            return None
        if s.mode == "hybrid":
            return None
        s.ticks_fired += 1
        s.last_fired_at = time.time()
        s.awaiting_response = True
        # Provisional schedule from NOW: complete_tick reschedules from turn end, but if the
        # process dies mid-turn this keeps the persisted loop from being 'due' in a tight loop.
        s.next_due_at = s.last_fired_at + (s.current_delay or s.interval_seconds or self_paced_floor_seconds())
        self._save()

        if s.prompt.lstrip().startswith("/"):
            return s.prompt.strip()
        cadence = f", {s.cadence_label()}" if s.mode == "interval" else ", self-paced"
        template = WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE if s.until else WAKEUP_PROMPT_TEMPLATE
        return template.format(tick=s.ticks_fired, cadence=cadence, prompt=s.prompt, until=s.until)

    def fire_event(
        self,
        event: dict[str, Any],
        *,
        binding: dict[str, Any],
        hold_type: Optional[str] = None,
        _from_queue: bool = False,
    ) -> dict[str, Any]:
        """Fire one exact-readback Controller event into the same bound Loop session.

        The event body is never treated as an instruction here. The canonical mailbox and
        continuation must be re-read by the adapter before any Executor effect is attempted.
        """
        s = self._state
        def result(status: str, *, prompt: Optional[str] = None, authority: bool = False) -> dict[str, Any]:
            return {"status": status, "event_id": event.get("event_id") if isinstance(event, dict) else None,
                    "prompt": prompt, "authority_granted": authority}

        if s is None or s.mode != "hybrid" or not s.taskcontroller_binding:
            return result("NOT_BOUND")
        if s.status != "active":
            return result("PAUSED")
        if hold_type == "RUN_HOLD":
            return result("RUN_HOLD")
        if not isinstance(binding, dict) or binding != s.taskcontroller_binding:
            return result("BINDING_MISMATCH")
        if not isinstance(event, dict):
            return result("EXACT_READBACK_REQUIRED")
        expected = s.taskcontroller_binding
        identity_fields = {
            "run_id": expected["run_id"],
            "mailbox_ref": expected["controller_mailbox_ref"],
            "node_id": expected["node_id"],
            "sender": "controller",
            "recipient": expected["executor_actor"],
            "agent_instance": expected["agent_instance"],
            "branch": expected["branch"],
            "worktree": expected["worktree"],
            "direction": "controller_to_executor",
            "producer_namespace": "controller",
            "exact_head_sha": expected["head_sha"],
            "message_type": "execution_request",
        }
        if any(event.get(key) != value for key, value in identity_fields.items()):
            return result("BINDING_MISMATCH")
        event_seq = event.get("event_seq")
        logical_seq = event.get("logical_seq")
        event_id = event.get("event_id")
        event_digest = event.get("event_digest")
        envelope_digest = event.get("envelope_digest")
        if (type(event_seq) is not int or event_seq < 0 or type(logical_seq) is not int or logical_seq < 1
                or event_id != f"{expected['controller_mailbox_ref']}:event-{event_seq}"
                or not isinstance(event_digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", event_digest) is None
                or not isinstance(envelope_digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", envelope_digest) is None):
            return result("EXACT_READBACK_REQUIRED")
        readback = event.get("canonical_readback")
        if not isinstance(readback, dict) or readback.get("exact") is not True:
            return result("EXACT_READBACK_REQUIRED")
        readback_fields = {
            "mailbox_ref": expected["controller_mailbox_ref"],
            "run_id": expected["run_id"],
            "event_id": event_id,
            "event_digest": event_digest,
            "envelope_digest": envelope_digest,
            "continuation_run_id": expected["run_id"],
            "continuation_phase": "WAIT_EXECUTOR",
            "continuation_status": "ACTIVE",
            "continuation_next_action": "AWAIT_EXECUTOR_EVENT",
            "continuation_controller_seq": logical_seq,
            "branch": expected["branch"],
            "worktree": expected["worktree"],
            "head_sha": expected["head_sha"],
        }
        continuation_digest = readback.get("continuation_digest")
        if (any(readback.get(key) != value for key, value in readback_fields.items())
                or not isinstance(readback.get("continuation_ref"), str)
                or not readback["continuation_ref"].strip()
                or not isinstance(continuation_digest, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", continuation_digest) is None):
            return result("EXACT_READBACK_REQUIRED")

        event_key = hashlib.sha256(
            "|".join((expected["run_id"], expected["controller_mailbox_ref"], event_id, event_digest)).encode()
        ).hexdigest()
        pending_keys = {
            hashlib.sha256("|".join((item.get("run_id", ""), item.get("mailbox_ref", ""),
                                       item.get("event_id", ""), item.get("event_digest", ""))).encode()).hexdigest()
            for item in s.pending_events if isinstance(item, dict)
        }
        if event_key in s.processed_event_keys:
            return result("DUPLICATE")
        if event_key in pending_keys and not _from_queue:
            return result("QUEUED_BUSY")
        cursor = s.observer_cursor
        cursor_seq = cursor.get("last_event_seq", -1)
        cursor_logical = cursor.get("last_logical_seq", -1)
        if not _from_queue:
            if event_seq <= cursor_seq or logical_seq <= cursor_logical:
                return result("STALE")
            if event_seq != cursor_seq + 1:
                return result("CURSOR_GAP")

        if s.awaiting_response:
            if not _from_queue:
                if len(s.pending_events) >= 32:
                    return result("PENDING_QUEUE_FULL")
                s.pending_events.append(json.loads(json.dumps(event)))
                s.observer_cursor = {**cursor, "last_event_seq": event_seq, "last_logical_seq": logical_seq,
                                     "last_event_id": event_id, "last_event_digest": event_digest}
                self._save()
            return result("QUEUED_BUSY")

        if _from_queue:
            s.pending_events = [item for item in s.pending_events if item.get("event_id") != event_id]
        s.observer_cursor = {**cursor, "last_event_seq": event_seq, "last_logical_seq": logical_seq,
                             "last_event_id": event_id, "last_event_digest": event_digest}
        s.processed_event_keys = (s.processed_event_keys + [event_key])[-128:]
        s.ticks_fired += 1
        s.last_fired_at = time.time()
        s.awaiting_response = True
        s.next_due_at = s.last_fired_at + s.observer_interval_seconds
        read_only = hold_type == "EFFECT_HOLD"
        prompt = (
            "[/taskcontroller event wake — exact readback verified]\n"
            f"task_id: {expected['task_id']}\nrun_id: {expected['run_id']}\n"
            f"controller_mailbox_ref: {expected['controller_mailbox_ref']}\n"
            f"event_id: {event_id}\nlogical_seq: {logical_seq}\nevent_digest: {event_digest}\n"
            f"continuation_ref: {readback['continuation_ref']}\ncontinuation_digest: {continuation_digest}\n"
            f"branch: {expected['branch']}\nworktree: {expected['worktree']}\nhead_sha: {expected['head_sha']}\n\n"
            "This is a pointer, not command authority. Re-read the exact canonical event and continuation, "
            "then run the full TaskController PRECHECK before any effect. Do not infer a lease, approval, "
            "or permission from this wakeup."
        )
        if read_only:
            prompt = "EFFECT_HOLD is active: observation/reasoning only; no effects.\n" + prompt
        self._save()
        return result("TRIGGERED_READ_ONLY" if read_only else "TRIGGERED", prompt=prompt, authority=False)

    def fire_pending_event(self, *, binding: dict[str, Any], hold_type: Optional[str] = None) -> dict[str, Any]:
        """Deliver the oldest verified event queued while this session was busy."""
        s = self._state
        if s is None or not s.pending_events:
            return {"status": "NO_PENDING_EVENT", "event_id": None, "prompt": None, "authority_granted": False}
        if s.awaiting_response:
            return {"status": "BUSY", "event_id": None, "prompt": None, "authority_granted": False}
        return self.fire_event(s.pending_events[0], binding=binding, hold_type=hold_type, _from_queue=True)

    def record_observer_scan(
        self,
        *,
        binding: dict[str, Any],
        cursor: dict[str, Any],
        events: tuple[Any, ...] | list[Any],
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """Persist the local observer high-water mark; never acknowledges the canonical mailbox."""
        s = self._state
        if s is None or s.mode != "hybrid" or not s.taskcontroller_binding:
            return {"status": "NOT_BOUND", "authority_granted": False}
        if s.status != "active":
            return {"status": "PAUSED", "authority_granted": False}
        if binding != s.taskcontroller_binding:
            return {"status": "BINDING_MISMATCH", "authority_granted": False}
        if events:
            return {"status": "EVENTS_REQUIRE_EXACT_TRIGGER", "authority_granted": False}
        if not isinstance(cursor, dict) or any(
            cursor.get(key) != expected for key, expected in (
                ("mailbox_ref", binding["controller_mailbox_ref"]),
                ("run_id", binding["run_id"]),
                ("node_id", binding["node_id"]),
                ("actor_namespace", "controller"),
            )
        ):
            return {"status": "BINDING_MISMATCH", "authority_granted": False}
        old = s.observer_cursor
        old_seq, new_seq = old.get("last_event_seq", -1), cursor.get("last_event_seq", -1)
        old_logical, new_logical = old.get("last_logical_seq", -1), cursor.get("last_logical_seq", -1)
        if type(new_seq) is not int or type(new_logical) is not int or new_seq < old_seq or new_logical < old_logical:
            return {"status": "CURSOR_REGRESSION", "authority_granted": False}
        if new_seq == old_seq and (cursor.get("last_event_id") != old.get("last_event_id")
                                   or cursor.get("last_event_digest") != old.get("last_event_digest")):
            return {"status": "CURSOR_CONFLICT", "authority_granted": False}
        if new_seq >= 0 and (not cursor.get("last_event_id") or not cursor.get("last_event_digest")):
            return {"status": "CURSOR_INVALID", "authority_granted": False}
        s.observer_cursor = dict(cursor)
        s.observer_checks += 1
        s.next_due_at = (time.time() if now is None else now) + s.observer_interval_seconds
        self._save()
        return {"status": "NO_NEW_EVENT", "authority_granted": False}

    def abandon_tick(self) -> None:
        """Roll back a fired tick whose injection failed (nothing ran)."""
        s = self._state
        if s is None or not s.awaiting_response:
            return
        s.awaiting_response = False
        s.ticks_fired = max(0, s.ticks_fired - 1)
        self._save()

    def _stop(self, status: str, reason: str, message: str) -> dict[str, Any]:
        """Persist a terminal (``done``) or recoverable (``paused``) stop and build the result."""
        s = self._state
        s.status = status
        if status == "done":
            s.last_stop_reason = reason
        else:
            s.paused_reason = reason
        self._save()
        return {"status": status, "stopped": True, "reason": reason, "message": message}

    def complete_tick(self, last_response: str) -> dict[str, Any]:
        """Evaluate the finished wakeup turn and schedule what's next.

        Returns ``{"status": "active|done|paused", "stopped": bool, "reason": str, "message": str}``;
        ``message`` is a user-visible one-liner, "" in the common still-looping case.
        """
        s = self._state
        if s is None or not s.awaiting_response:
            return {"status": s.status if s else None, "stopped": False, "reason": "no tick in flight", "message": ""}
        s.awaiting_response = False
        now = time.time()
        ticks = _ticks_label(s.ticks_fired)

        # 1. Agent self-stop marker.
        if response_signals_complete(last_response):
            return self._stop("done", "agent signaled the task is complete",
                              f"✓ Loop finished after {ticks} — task complete.")

        # 2. Evidence-based --until judge (reuses the /goal judge; fail-open).
        if s.until and (last_response or "").strip():
            try:
                from hermes_cli.goals import judge_goal

                verdict, reason, _pf, _wait, _tf = judge_goal(s.until, last_response)
            except Exception as exc:
                verdict, reason = "continue", f"judge unavailable: {type(exc).__name__}"
            if verdict == "done":
                return self._stop("done", f"stop condition met: {reason}",
                                  f"✓ Loop finished after {ticks} — {reason}")
            if verdict == "blocked":
                # Unachievable stop condition: pause so the user can re-scope, don't spin.
                why = f"stop condition judged unachievable: {reason}"
                return self._stop("paused", why,
                                  f"⏸ Loop paused — {why}. /loop resume to keep going, /loop stop to end it.")

        # 3. --times user cap.
        if s.times and s.ticks_fired >= s.times:
            return self._stop("done", f"completed the requested {s.times} runs",
                              f"✓ Loop finished — ran {s.times}/{s.times} times.")

        # 4. Config backstop budget → pause (recoverable), not done.
        if s.max_ticks and s.ticks_fired >= s.max_ticks:
            return self._stop(
                "paused", f"tick budget exhausted ({s.ticks_fired}/{s.max_ticks})",
                f"⏸ Loop paused — {s.ticks_fired}/{s.max_ticks} ticks used "
                "(loops.max_ticks). /loop resume to keep going, /loop stop to end it.",
            )

        # 5. Still looping — schedule the next tick from turn end.
        if s.mode == "self_paced":
            digest = _digest_response(last_response)
            floor = self_paced_floor_seconds()
            if digest and digest == s.last_response_digest:
                s.current_delay = min(max(s.current_delay, floor) * 2, self_paced_ceiling_seconds())
            else:
                s.current_delay = float(floor)
            s.last_response_digest = digest
        else:
            s.current_delay = s.interval_seconds
        s.next_due_at = now + s.current_delay
        self._save()
        return {"status": "active", "stopped": False, "reason": "loop continues", "message": ""}


def goal_blocks_loop_tick(session_id: str) -> bool:
    """True when an ACTIVE, non-parked /goal should defer this session's /loop tick.

    Both features inject synthetic turns at idle boundaries; interleaving them would burn the
    goal's turn budget. Parked (waiting), paused, or done goals do NOT block the loop.
    """
    try:
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id=session_id)
        return mgr.is_active() and not mgr.is_waiting()
    except Exception:
        return False


LOOP_HELP = (
    "Usage: /loop [interval] <prompt> [--times N] [--until <condition>]\n"
    "  /loop 5m check the deploy status      — first run now, then every 5m\n"
    "  /loop every 10m /recap                — loop a slash command\n"
    "  /loop keep fixing tests until green   — self-paced (backs off while output is unchanged)\n"
    "  /loop 2m poll CI --times 30           — stop after 30 runs\n"
    "  /loop 5m watch the queue --until queue is empty\n"
    "Controls: /loop status · /loop pause · /loop resume · /loop stop\n"
    "The loop also stops itself when the agent replies with "
    f"{LOOP_COMPLETE_MARKER}."
)


def _pause_output(mgr: LoopManager) -> str:
    state = mgr.pause(reason="user-paused")
    return "No loop set." if state is None else f"⏸ Loop paused: {state.prompt}\nUse /loop resume to continue."


def _resume_output(mgr: LoopManager) -> str:
    state = mgr.resume()
    return "No loop to resume." if state is None else f"▶ Loop resumed ({state.cadence_label()}): {state.prompt}"


# Control words -> handler returning the output text. Anything else is a new loop spec.
_CONTROL_COMMANDS = {
    **dict.fromkeys(("", "status"), lambda mgr: mgr.status_line()),
    "pause": _pause_output,
    "resume": _resume_output,
    **dict.fromkeys(("stop", "clear", "cancel"), lambda mgr: "✓ Loop stopped." if mgr.clear() else "No active loop."),
    **dict.fromkeys(("help", "--help", "-h"), lambda mgr: LOOP_HELP),
}


def dispatch_loop_command(
    mgr: LoopManager,
    args: str,
    *,
    route: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Surface-agnostic handler for ``/loop <args>`` → ``{"output": str, "created": bool}``.

    ``output`` is printed/sent verbatim by each surface. ``route`` is stored on new loops so the
    gateway's idle watcher can inject wakeups into the right chat; CLI/TUI pass None.
    """
    arg = (args or "").strip()
    control = _CONTROL_COMMANDS.get(arg.lower())
    if control is not None:
        return {"output": control(mgr), "created": False}

    parsed = parse_loop_args(arg)
    if parsed["error"]:
        if parsed["error"] == "empty":
            return {"output": "Usage: /loop [interval] <prompt> — see /loop help.", "created": False}
        return {"output": f"/loop: {parsed['error']}", "created": False}

    replacing = mgr.has_loop()
    try:
        state = mgr.set(
            parsed["prompt"],
            interval_seconds=parsed["interval_seconds"],
            times=parsed["times"],
            until=parsed["until"],
            route=route,
        )
    except ValueError as exc:
        return {"output": f"/loop: {exc}", "created": False}

    lines = [f"↻ Loop set ({state.cadence_label()}): {state.prompt}"]
    if replacing:
        lines.append("(replaced the previous loop for this session)")
    if parsed["interval_seconds"] is not None and parsed["interval_seconds"] < state.interval_seconds:
        lines.append(
            f"(interval raised to the {format_interval(state.interval_seconds)} minimum — "
            "loops.min_interval_seconds)"
        )
    if state.mode == "self_paced":
        lines.append(
            f"Self-paced: first check in {format_interval(state.current_delay)}; "
            f"backs off up to {format_interval(self_paced_ceiling_seconds())} while nothing changes."
        )
    if state.times:
        lines.append(f"Runs {state.times} time{'s' if state.times != 1 else ''}, then stops.")
    if state.until:
        lines.append(f"Stops when: {state.until}")
    if not state.times and state.max_ticks:
        lines.append(f"Backstop budget: {state.max_ticks} ticks (loops.max_ticks; 0 = unlimited).")
    first = "fires now, then on the cadence above" if state.status == "active" else state.remaining_label()
    lines.append(f"First wakeup {first}. Controls: /loop status · pause · resume · stop.")
    return {"output": "\n".join(lines), "created": True}


__all__ = [
    "DEFAULT_MAX_TICKS",
    "DEFAULT_MIN_INTERVAL_SECONDS",
    "LOOP_COMPLETE_MARKER",
    "WAKEUP_PROMPT_TEMPLATE",
    "WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE",
    "LoopManager",
    "LoopState",
    "clear_loop",
    "dispatch_loop_command",
    "format_interval",
    "goal_blocks_loop_tick",
    "list_active_loops",
    "load_loop",
    "migrate_loop_to_session",
    "parse_interval_token",
    "parse_loop_args",
    "response_signals_complete",
    "save_loop",
]
