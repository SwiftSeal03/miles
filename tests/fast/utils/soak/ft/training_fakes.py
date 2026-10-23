import asyncio
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tests.fast.utils.soak.soak_fakes import _cell_target
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakObservationEvent

from miles.utils.async_utils import Disposer


@dataclass
class _TrainingHarness:
    log: EventLog
    calls: list[str] = field(default_factory=list)
    failure: str | None = None
    publish_task: asyncio.Task[None] | None = None

    async def train(self, args: Namespace, *, disposer: Disposer) -> None:
        self.calls.append("async_train" if args.fully_async else "sync_train")
        disposer.add(self.dispose)
        if self.failure == "train":
            raise RuntimeError("training failed")

    async def prepare_eval(self) -> None:
        self.calls.append("resume")
        if self.failure == "resume":
            await asyncio.Event().wait()
        self.publish_task = asyncio.create_task(self._publish())

    async def _publish(self) -> None:
        now = datetime.now(timezone.utc)
        self.log.append(
            SoakObservationEvent(timestamp=now, completed_at=now, targets=[_cell_target(kind="rollout")])
        )
        self.calls.append("observed_ready")

    async def dispose(self) -> None:
        self.calls.append("dispose")
        if self.failure == "cleanup":
            raise RuntimeError("cleanup failed")
