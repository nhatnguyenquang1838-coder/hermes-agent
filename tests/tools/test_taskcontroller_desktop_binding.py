"""TaskController live delivery binds to an explicit Desktop session, not a title."""
from __future__ import annotations

import json

import pytest


def _binding(*, run_id="test-scrum781-run", agent_instance="hermes-desktop-instance-1"):
    return {
        "task_id": "SCRUM-781",
        "run_id": run_id,
        "repository": "example/gwc",
        "branch": "test/SCRUM-781",
        "worktree": "worktrees/gwc/SCRUM-781-test-run",
        "head_sha": "a" * 40,
        "session_id": "executor-session",
        "executor_actor": "hermes-desktop",
        "agent_instance": agent_instance,
        "capability": "taskcontroller.executor",
        "observer_role": "MAILBOX_OBSERVER",
        "controller_mailbox_ref": "github://example/gwc/issues/1#test-scrum781-controller-v2",
        "executor_mailbox_ref": "github://example/gwc/issues/1#test-scrum781-executor-v2",
        "mailbox_ref": "github://example/gwc/issues/1#test-scrum781-controller-v2",
        "wakeup_binding": "hermes-desktop-local-ipc",
    }


def _live_owner(tmp_path, binding):
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="executor-session", source="cli")
    db.set_session_title("executor-session", "SCRUM-781 Executor")
    lease, refusal = try_acquire_active_session(
        session_id="executor-session",
        surface="desktop",
        config={},
        registry_home=tmp_path,
        metadata={
            "live_session_id": "desktop-live-1",
            "bot_live_delivery_consumer": True,
            "taskcontroller_binding": binding,
        },
    )
    assert refusal is None and lease is not None
    return db, lease


def test_explicit_taskcontroller_owner_does_not_require_bot_chat_title(tmp_path):
    from tools import bot_live_delivery as delivery

    binding = _binding()
    db, lease = _live_owner(tmp_path, binding)
    try:
        owner = delivery.find_taskcontroller_live_owner(
            tmp_path,
            run_id=binding["run_id"],
            agent_instance=binding["agent_instance"],
            wakeup_binding=binding["wakeup_binding"],
        )
        assert owner is not None
        assert owner["session_id"] == "executor-session"
        assert owner["lease_id"] == lease.lease_id
        assert owner["live_session_id"] == "desktop-live-1"
        assert owner["taskcontroller_binding"] == binding
    finally:
        lease.release()
        db.close()


@pytest.mark.parametrize(
    "lookup",
    [
        {"run_id": "another-run"},
        {"agent_instance": "another-desktop"},
        {"wakeup_binding": "github-actions-consumer"},
    ],
)
def test_taskcontroller_owner_fails_closed_on_binding_mismatch(tmp_path, lookup):
    from tools import bot_live_delivery as delivery

    binding = _binding()
    db, lease = _live_owner(tmp_path, binding)
    try:
        expected = {
            "run_id": binding["run_id"],
            "agent_instance": binding["agent_instance"],
            "wakeup_binding": binding["wakeup_binding"],
        }
        expected.update(lookup)
        assert delivery.find_taskcontroller_live_owner(tmp_path, **expected) is None
    finally:
        lease.release()
        db.close()


def test_delivery_ticket_pins_taskcontroller_binding_and_wakeup_category(tmp_path):
    from tools import bot_live_delivery as delivery

    owner = {
        "profile_home": str(tmp_path.resolve()),
        "session_id": "executor-session",
        "lease_id": "lease-781",
        "live_session_id": "desktop-live-1",
        "taskcontroller_binding": _binding(),
    }
    pointer = {
        "protocol": "dw.taskcontroller.wakeup/v1",
        "run_id": owner["taskcontroller_binding"]["run_id"],
        "sender": "controller",
        "recipient": owner["taskcontroller_binding"]["executor_actor"],
        "mailbox_ref": owner["taskcontroller_binding"]["mailbox_ref"],
        "seq": 4,
        "updated_at": "2026-10-12T00:00:00Z",
    }
    record = delivery.deliver_to_live_owner(
        tmp_path,
        owner,
        json.dumps(pointer, sort_keys=True),
        delivery_id="a" * 32,
        notification_category="taskcontroller-wakeup",
    )
    assert record["notification_category"] == "taskcontroller-wakeup"
    assert record["owner"]["taskcontroller_binding"] == owner["taskcontroller_binding"]
    assert "command" not in json.loads(record["message"])


def test_old_run_delivery_cannot_be_claimed_by_rebound_session(tmp_path):
    from tools import bot_live_delivery as delivery

    old_binding = _binding(run_id="old-run")
    current_binding = _binding(run_id="current-run")
    old_owner = {
        "profile_home": str(tmp_path.resolve()),
        "session_id": "executor-session",
        "lease_id": "lease-781",
        "live_session_id": "desktop-live-1",
        "taskcontroller_binding": old_binding,
    }
    current_owner = {**old_owner, "taskcontroller_binding": current_binding}
    pointer = json.dumps({
        "protocol": "dw.taskcontroller.wakeup/v1",
        "run_id": old_binding["run_id"],
        "sender": "controller",
        "recipient": old_binding["executor_actor"],
        "mailbox_ref": old_binding["mailbox_ref"],
        "seq": 4,
        "updated_at": "2026-10-12T00:00:00Z",
    }, sort_keys=True)
    ticket = delivery.deliver_to_live_owner(
        tmp_path, old_owner, pointer, delivery_id="b" * 32,
        notification_category="taskcontroller-wakeup",
    )

    assert delivery.claim_pending_delivery(tmp_path, current_owner) is None
    claimed = delivery.claim_pending_delivery(tmp_path, old_owner)
    assert claimed is not None
    assert claimed["delivery_id"] == ticket["delivery_id"]
    assert claimed["owner"]["taskcontroller_binding"] == old_binding
