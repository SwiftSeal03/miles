import asyncio
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from tests.fast.utils.soak.soak_fakes import _cell_target
from tests.fast.utils.soak.ft.training_fakes import _TrainingHarness
from tests.utils.soak.ft import training
from miles.utils.async_utils import with_disposer
from miles.utils.function_registry import function_registry
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent, StoredEvent
from tests.utils.soak.ft.training import _has_ready_observation, _wait_for_ready_observation


class TestTerminalReadiness:
    def test_only_a_new_ready_observation_releases_training(self, events: list[SoakEvent]) -> None:
        """Paused health and an earlier ready sample cannot authorize teardown."""
        before = datetime.now(timezone.utc)
        events.append(
            SoakObservationEvent(
                timestamp=before - timedelta(seconds=1),
                completed_at=before,
                targets=[_cell_target(kind="rollout")],
            )
        )
        assert not _has_ready_observation(events=events, not_before=before)
        events.append(
            SoakObservationEvent(
                timestamp=before,
                completed_at=before,
                targets=[_cell_target(kind="rollout", ready=False)],
            )
        )
        assert not _has_ready_observation(events=events, not_before=before)
        events.append(
            SoakObservationEvent(timestamp=before, completed_at=before, targets=[_cell_target(kind="rollout")])
        )
        assert _has_ready_observation(events=events, not_before=before)

    @pytest.mark.parametrize("failure", ["missing", "dead", "errors", "admission", "stale", "later_unready"])
    def test_incomplete_or_outdated_confirmation_does_not_release_training(
        self, events: list[SoakEvent], failure: str
    ) -> None:
        """Every confirmation must retain the original final-state requirements."""
        now = datetime.now(timezone.utc)
        target = _cell_target(kind="rollout", alive=failure != "dead")
        observation = SoakObservationEvent(
            timestamp=now,
            completed_at=now,
            targets=[] if failure == "missing" else [target],
            errors={"cells": "failed"} if failure == "errors" else {},
        )
        if failure == "admission":
            events.pop()
        if failure == "stale":
            observation = observation.model_copy(update={"timestamp": now - timedelta(seconds=61)})
        events.append(observation)
        if failure == "later_unready":
            events.append(observation.model_copy(update={"targets": [_cell_target(kind="rollout", ready=False)]}))
        assert not _has_ready_observation(events=events, not_before=now - timedelta(seconds=120))

    async def test_a_partial_last_record_waits_until_the_outer_writer_completes_it(
        self, events: list[SoakEvent], tmp_path: Path
    ) -> None:
        """An in-flight JSONL write must not fail parsing or count as confirmation."""
        path = tmp_path / "events.jsonl"
        log = EventLog(path)
        for event in events:
            log.append(event)
        now = datetime.now(timezone.utc)
        record = StoredEvent(
            sequence=len(events),
            event=SoakObservationEvent(timestamp=now, completed_at=now, targets=[_cell_target(kind="rollout")]),
        ).model_dump_json()
        with path.open("a") as stream:
            stream.write(record[:20])
        waiting = asyncio.create_task(_wait_for_ready_observation(path=path, not_before=now))
        try:
            await asyncio.sleep(0)
            assert not waiting.done()
            with path.open("a") as stream:
                stream.write(record[20:] + "\n")
            await asyncio.wait_for(waiting, timeout=3)
        finally:
            waiting.cancel()

    async def test_a_complete_corrupt_record_is_not_silently_ignored(
        self, events: list[SoakEvent], tmp_path: Path
    ) -> None:
        """Only incomplete trailing records may be deferred."""
        path = tmp_path / "events.jsonl"
        log = EventLog(path)
        for event in events:
            log.append(event)
        with path.open("a") as stream:
            stream.write("{bad json}\n")
        with pytest.raises(ValueError):
            await _wait_for_ready_observation(path=path, not_before=datetime.now(timezone.utc))


class TestTrainingLifecycle:
    @pytest.mark.parametrize("fully_async", [False, True])
    @pytest.mark.parametrize("failure", [None, "train", "cleanup", "resume"])
    async def test_readiness_precedes_cleanup_and_failures_remain_failures(
        self, events: list[SoakEvent], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        fully_async: bool, failure: str | None,
    ) -> None:
        """Both drivers must observe readiness before cleanup and propagate unsuccessful execution."""
        args = Namespace(cluster_backend="ray", fully_async=fully_async, save_debug_event_data=str(tmp_path / "training"))
        log = EventLog(tmp_path / "events.jsonl")
        for index, event in enumerate(events):
            log.append(event.model_copy(update={"sources": {"training_events": Path(args.save_debug_event_data)}}) if index == 0 else event)
        harness = _TrainingHarness(log=log, failure=failure)
        monkeypatch.setenv("MILES_SOAK_EVENT_LOG", str(log.path))
        monkeypatch.setattr(training, "get_backend_capability", lambda args: None)
        monkeypatch.setattr(training, "create_inference_controller_handle", lambda *, capability: harness)
        if failure == "resume":
            monkeypatch.setattr(training, "_TERMINAL_TIMEOUT_SECONDS", 0.01)
        module = "train_async.train" if fully_async else "train.train"
        with function_registry.temporary(module, harness.train):
            if failure is None:
                await with_disposer(training._train, args)
            else:
                expected = TimeoutError if failure == "resume" else RuntimeError
                with pytest.raises(expected):
                    await with_disposer(training._train, args)
        assert harness.calls[0] == ("async_train" if fully_async else "sync_train")
        assert harness.calls[-1] == "dispose"
        if failure == "train":
            assert "resume" not in harness.calls
        else:
            assert harness.calls.count("resume") == 1
        if failure in (None, "cleanup"):
            assert harness.calls.index("observed_ready") < harness.calls.index("dispose")
