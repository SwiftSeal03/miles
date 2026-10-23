import threading
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.distributed as dist
from tests.fast.dist_utils import init_gloo
from torchft.process_group import ProcessGroupGloo

from miles.utils.dumper_utils import _barrier_after_dump_dir_cleanup


def _run_missing_dump_peer(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank=rank, world_size=world_size, port=port)
    dump_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=2))
    wrapped = ProcessGroupGloo(pg=dump_group)
    nccl_group = MagicMock()
    state = SimpleNamespace(
        indep_dp=SimpleNamespace(rank=rank, size=world_size, group=nccl_group, gloo_group=wrapped, debug_info={})
    )
    dist.barrier()
    try:
        if rank == 0:
            heartbeat_times: list[float] = []
            started_at = time.monotonic()
            timer = threading.Timer(0.5, lambda: heartbeat_times.append(time.monotonic()))
            timer.start()
            try:
                with (
                    patch("miles.utils.dumper_utils.get_parallel_state", return_value=state),
                    patch("miles.utils.dumper_utils.dist.is_initialized", return_value=False),
                    patch("miles.utils.dumper_utils.logger") as logger,
                ):
                    _barrier_after_dump_dir_cleanup()
                assert heartbeat_times, "Heartbeat could not run during the dump barrier"
                assert heartbeat_times[0] - started_at < 1.5, "Heartbeat was delayed until barrier timeout"
                assert logger.error.call_count == 1
                nccl_group.barrier.assert_not_called()
            finally:
                timer.cancel()
                timer.join(timeout=5)
        dist.monitored_barrier(timeout=timedelta(seconds=10))
    finally:
        dist.destroy_process_group(dump_group)
        dist.destroy_process_group()
