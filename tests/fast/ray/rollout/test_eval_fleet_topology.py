import argparse
import asyncio
from types import SimpleNamespace
from miles.ray.rollout.eval_fleet import InferenceControllerEvalFleet


class TestEvalFleetTopology:

    async def test_empty_fleet_is_reported_without_inventing_a_configured_engine(self) -> None:
        """An observed empty fleet stays empty regardless of configured deployment size."""
        server = SimpleNamespace(context_lock=asyncio.Lock(), engine_gpu_counts=[], router_ip='127.0.0.1', router_port=8000)
        fleet = InferenceControllerEvalFleet(argparse.Namespace(eval_num_gpus=16), srv=server)
        assert (await fleet.info()).engine_gpu_counts == []
