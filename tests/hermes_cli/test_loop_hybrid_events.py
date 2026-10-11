"""Contract tests for TaskController-driven /loop event wakes.

The event body is never carried by the notification: only the canonical
mailbox adapter may produce an exact-readback observation for a bound run.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli import goals

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture
def binding():
    return {
        "task_id": "SCRUM-781",
        "run_id": "test-scrum781-run",
        "node_id": "SCRUM-781",
        "repository": "example/gwc",
        "branch": "test/SCRUM-781",
        "worktree": "worktrees/gwc/SCRUM-781-test-run",
        "head_sha": "a" * 40,
        "session_id": "desktop-session-1",
        "executor_actor": "hermes-desktop",
        "agent_instance": "hermes-desktop-instance-1",
        "capability": "taskcontroller.executor",
        "observer_role": "MAILBOX_OBSERVER",
        "controller_mailbox_ref": "github://example/gwc/issues/1#test-scrum781-controller-v2",
        "executor_mailbox_ref": "github://example/gwc/issues/1#test-scrum781-executor-v2",
        "mailbox_ref": "github://example/gwc/issues/1#test-scrum781-controller-v2",
        "wakeup_binding": "hermes-desktop-local-ipc",
    }


@pytest.fixture
def observed_event(binding):
    event_id = f"{binding['mailbox_ref']}:event-3"
    event_digest = "sha256:" + "a" * 64
    envelope_digest = "sha256:" + "b" * 64
    return {
        "run_id": binding["run_id"],
        "mailbox_ref": binding["mailbox_ref"],
        "node_id": binding["node_id"],
        "sender": "controller",
        "recipient": binding["executor_actor"],
        "agent_instance": binding["agent_instance"],
        "branch": binding["branch"],
        "worktree": binding["worktree"],
        "direction": "controller_to_executor",
        "producer_namespace": "controller",
        "event_seq": 3,
        "logical_seq": 3,
        "event_id": event_id,
        "event_digest": event_digest,
        "envelope_digest": envelope_digest,
        "exact_head_sha": binding["head_sha"],
        "message_type": "execution_request",
        "canonical_readback": {
            "exact": True,
            "mailbox_ref": binding["mailbox_ref"],
            "run_id": binding["run_id"],
            "event_id": event_id,
            "event_digest": event_digest,
            "envelope_digest": envelope_digest,
            "continuation_run_id": binding["run_id"],
            "continuation_ref": "github://example/gwc/issues/1#issuecomment-42",
            "continuation_digest": "sha256:" + "d" * 64,
            "continuation_phase": "WAIT_EXECUTOR",
            "continuation_status": "ACTIVE",
            "continuation_next_action": "AWAIT_EXECUTOR_EVENT",
            "continuation_controller_seq": 3,
            "branch": binding["branch"],
            "worktree": binding["worktree"],
            "head_sha": binding["head_sha"],
        },
    }


def _cursor(binding, *, event_seq=2, logical_seq=2, event_id=None):
    return {
        "protocol": "dw.taskcontroller.mailbox-cursor/v1",
        "mailbox_ref": binding["mailbox_ref"],
        "run_id": binding["run_id"],
        "node_id": "SCRUM-781",
        "actor_namespace": "controller",
        "last_logical_seq": logical_seq,
        "last_event_seq": event_seq,
        "last_event_id": event_id or f"{binding['mailbox_ref']}:event-{event_seq}",
        "last_event_digest": "sha256:" + "c" * 64,
    }


def test_hybrid_loop_persists_binding_and_observer_cursor(hermes_home, binding):
    from hermes_cli.loops import LoopManager, load_loop

    manager = LoopManager(session_id=binding["session_id"])
    state = manager.set_hybrid(
        "Continue the exact authorized SCRUM-781 work",
        binding=binding,
        observer_interval_seconds=60,
        observer_cursor=_cursor(binding),
    )

    loaded = load_loop(binding["session_id"])
    assert state.mode == "hybrid"
    assert loaded is not None
    assert loaded.taskcontroller_binding == binding
    assert loaded.observer_cursor == _cursor(binding)
    assert loaded.observer_interval_seconds == 60


def test_hybrid_timer_is_read_only_observer_due_not_an_llm_tick(hermes_home, binding):
    from hermes_cli.loops import LoopManager

    manager = LoopManager(session_id=binding["session_id"])
    state = manager.set_hybrid("observe SCRUM-781", binding=binding, observer_interval_seconds=60)
    state.next_due_at = time.time() - 1

    assert manager.is_due() is True  # observer scan may run
    assert manager.fire_tick() is None  # timer alone must never create an LLM turn
    assert manager.state.ticks_fired == 0
    assert manager.state.awaiting_response is False


def test_only_exact_readback_event_triggers_the_same_bound_session(
    hermes_home, binding, observed_event
):
    from hermes_cli.loops import LoopManager

    manager = LoopManager(session_id=binding["session_id"])
    manager.set_hybrid("continue bounded work", binding=binding, observer_interval_seconds=60, observer_cursor=_cursor(binding))

    result = manager.fire_event(observed_event, binding=binding)

    assert result["status"] == "TRIGGERED"
    assert result["event_id"] == observed_event["event_id"]
    assert manager.state.awaiting_response is True
    assert manager.state.ticks_fired == 1
    assert observed_event["mailbox_ref"] in result["prompt"]


def test_push_and_poll_of_the_same_event_are_one_logical_trigger(
    hermes_home, binding, observed_event
):
    from hermes_cli.loops import LoopManager

    manager = LoopManager(session_id=binding["session_id"])
    manager.set_hybrid("continue bounded work", binding=binding, observer_interval_seconds=60, observer_cursor=_cursor(binding))

    pushed = manager.fire_event(observed_event, binding=binding)
    polled = manager.fire_event(dict(observed_event), binding=binding)

    assert pushed["status"] == "TRIGGERED"
    assert polled["status"] == "DUPLICATE"
    assert polled["prompt"] is None
    assert manager.state.ticks_fired == 1


@pytest.mark.parametrize(
    "mismatch",
    [
        {"run_id": "other-run"},
        {"mailbox_ref": "github://example/gwc/issues/1#other-mailbox"},
        {"agent_instance": "other-desktop-instance"},
        {"branch": "test/OTHER"},
        {"worktree": "worktrees/gwc/OTHER"},
        {"exact_head_sha": "f" * 40},
    ],
)
def test_event_must_match_run_mailbox_agent_branch_worktree_and_head(
    hermes_home, binding, observed_event, mismatch
):
    from hermes_cli.loops import LoopManager

    manager = LoopManager(session_id=binding["session_id"])
    manager.set_hybrid("continue bounded work", binding=binding, observer_interval_seconds=60, observer_cursor=_cursor(binding))
    mismatched = dict(observed_event, **mismatch)

    result = manager.fire_event(mismatched, binding=binding)

    assert result["status"] == "BINDING_MISMATCH"
    assert result["prompt"] is None
    assert manager.state.awaiting_response is False
    assert manager.state.ticks_fired == 0


def test_unqualified_observer_and_noncanonical_pointer_fail_closed(
    hermes_home, binding, observed_event
):
    from hermes_cli.loops import LoopManager

    manager = LoopManager(session_id=binding["session_id"])
    unqualified = dict(binding, observer_role="MAILBOX_POLL")
    with pytest.raises(ValueError, match="MAILBOX_OBSERVER"):
        manager.set_hybrid("observe", binding=unqualified, observer_interval_seconds=60)

    manager.set_hybrid("observe", binding=binding, observer_interval_seconds=60)
    no_digest = dict(observed_event, event_digest="")
    result = manager.fire_event(no_digest, binding=binding)
    assert result["status"] == "EXACT_READBACK_REQUIRED"
    assert result["prompt"] is None
    assert manager.state.ticks_fired == 0


def test_pause_and_run_hold_block_observation_and_event_wakes(
    hermes_home, binding, observed_event
):
    from hermes_cli.loops import LoopManager

    manager = LoopManager(session_id=binding["session_id"])
    manager.set_hybrid("continue bounded work", binding=binding, observer_interval_seconds=60, observer_cursor=_cursor(binding))
    manager.pause("user-paused")
    assert manager.fire_event(observed_event, binding=binding)["status"] == "PAUSED"
    assert manager.is_due() is False

    manager.resume()
    assert manager.fire_event(observed_event, binding=binding, hold_type="RUN_HOLD")["status"] == "RUN_HOLD"
    assert manager.state.ticks_fired == 0


def test_effect_hold_can_trigger_observation_but_never_grants_effect_authority(
    hermes_home, binding, observed_event
):
    from hermes_cli.loops import LoopManager

    manager = LoopManager(session_id=binding["session_id"])
    manager.set_hybrid("continue bounded work", binding=binding, observer_interval_seconds=60, observer_cursor=_cursor(binding))

    result = manager.fire_event(observed_event, binding=binding, hold_type="EFFECT_HOLD")

    assert result["status"] == "TRIGGERED_READ_ONLY"
    assert "EFFECT_HOLD" in result["prompt"]
    assert result["authority_granted"] is False


def test_observer_cursor_advances_without_llm_turn_when_no_new_event(hermes_home, binding):
    from hermes_cli.loops import LoopManager

    manager = LoopManager(session_id=binding["session_id"])
    manager.set_hybrid("observe SCRUM-781", binding=binding, observer_interval_seconds=60)

    result = manager.record_observer_scan(
        binding=binding,
        cursor=_cursor(binding, event_seq=3, logical_seq=3),
        events=(),
        now=time.time(),
    )

    assert result["status"] == "NO_NEW_EVENT"
    assert manager.state.observer_cursor["last_event_seq"] == 3
    assert manager.state.ticks_fired == 0
    assert manager.state.awaiting_response is False


def test_event_received_while_busy_is_persisted_without_second_turn(
    hermes_home, binding, observed_event
):
    from hermes_cli.loops import LoopManager

    manager = LoopManager(session_id=binding["session_id"])
    manager.set_hybrid("continue bounded work", binding=binding, observer_interval_seconds=60, observer_cursor=_cursor(binding))
    manager.state.awaiting_response = True

    result = manager.fire_event(observed_event, binding=binding)

    assert result["status"] == "QUEUED_BUSY"
    assert result["prompt"] is None
    assert manager.state.pending_events == [observed_event]
    assert manager.state.ticks_fired == 0


def test_event_queue_and_observed_cursor_survive_loop_manager_restart(
    hermes_home, binding, observed_event
):
    from hermes_cli.loops import LoopManager

    manager = LoopManager(session_id=binding["session_id"])
    manager.set_hybrid(
        "continue bounded work",
        binding=binding,
        observer_interval_seconds=60,
        observer_cursor=_cursor(binding),
    )
    manager.state.awaiting_response = True
    manager.fire_event(observed_event, binding=binding)

    restarted = LoopManager(session_id=binding["session_id"])
    assert restarted.state is not None
    assert restarted.state.observer_cursor["last_event_seq"] == 3
    assert restarted.state.pending_events == [observed_event]
    assert restarted.state.taskcontroller_binding == binding
