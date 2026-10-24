import asyncio
import logging
import os
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from tests.utils.soak.core.checkers.end_state import assert_end_state_complete
from tests.utils.soak.core.events import SoakEvent, SoakRunContextEvent, StoredEvent
from tests.utils.soak.core.views import admission_closed, latest_observation, project_actions, tail_started_at

from miles.ray.specs.inference import create_inference_controller_handle
from miles.ray.wiring import get_backend_capability
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig
from miles.utils.function_registry import load_function
from miles.utils.workers.types import ClusterBackend

logger = logging.getLogger(__name__)

_EVENT_LOG_ENV = "MILES_SOAK_EVENT_LOG"
_OBSERVATION_MAX_AGE_SECONDS = 60.0


async def _train(args: Namespace, *, disposer: Disposer) -> None:
    assert args.fully_async and ClusterBackend(args.cluster_backend) is ClusterBackend.RAY
    path = Path(os.environ[_EVENT_LOG_ENV])
    with path.open() as stream:
        context = StoredEvent.model_validate_json(stream.readline()).event
    assert isinstance(context, SoakRunContextEvent), "Missing soak run context"
    assert context.sources["training_events"] == Path(args.save_debug_event_data), "Wrong soak evidence path"

    train = load_function("train_async.train")
    await train(args, disposer=disposer)

    async with asyncio.timeout(_terminal_timeout(args)):
        controller = create_inference_controller_handle(capability=get_backend_capability(args))
        await controller.prepare_eval()
        resumed_at = datetime.now(timezone.utc)
        logger.info("soak op=terminal_readiness phase=start resumed_at=%s path=%s", resumed_at.isoformat(), path)
        await _wait_for_ready_observation(path=path, not_before=resumed_at)


def _terminal_timeout(args: Namespace) -> float:
    configs = [
        SimpleHealthCheckerConfig.from_args(args, prefix=prefix)
        for prefix in ("trainer_heartbeat_checker", "rollout_health_check")
    ]
    return max(2 * c.first_wait + 2 * c.interval + c.timeout for c in configs) + _OBSERVATION_MAX_AGE_SECONDS


async def _wait_for_ready_observation(*, path: Path, not_before: datetime) -> None:
    events: list[SoakEvent] = []
    with path.open() as stream:
        while True:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line.endswith("\n"):
                    stream.seek(position)
                    break
                stored = StoredEvent.model_validate_json(line)
                assert stored.sequence == len(events), "Missing or reordered soak event"
                events.append(stored.event)

            if _has_ready_observation(events=events, not_before=not_before):
                observation = latest_observation(events)
                assert observation is not None
                logger.info("soak op=terminal_readiness phase=end observation=%s", observation.model_dump_json())
                return
            await asyncio.sleep(1.0)


def _has_ready_observation(*, events: list[SoakEvent], not_before: datetime) -> bool:
    assert events and isinstance(events[0], SoakRunContextEvent), "Missing soak run context"
    context = events[0].context
    if admission_closed(events) is None:
        return False
    for request_id, action in project_actions(events).items():
        if action.result is None:
            return False
        assert action.result.returned, f"Action failed before terminal readiness: {request_id}"

    observation = latest_observation(events)
    if observation is None or observation.completed_at is None:
        return False
    if observation.timestamp < max(not_before, tail_started_at(events)):
        return False
    now = datetime.now(timezone.utc)
    assert observation.timestamp <= observation.completed_at <= now, "Invalid terminal observation times"
    if (now - observation.timestamp).total_seconds() > _OBSERVATION_MAX_AGE_SECONDS:
        return False

    try:
        assert_end_state_complete(
            events,
            expected_count_of_kind={
                kind: config.expected_count for kind, config in context.config.target_configs.items()
            },
            observation_ends_with_sut=False,
        )
    except AssertionError:
        return False
    return True


if __name__ == "__main__":
    asyncio.run(with_disposer(_train, parse_args()))
