import sys
from argparse import Namespace
from pathlib import Path

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.ray.specs.inference import InferenceEngineSpec


@pytest.fixture(autouse=True)
def _interpreter_without_runner_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-m", "pytest"])


def _make_args(**overrides: object) -> Namespace:
    args = make_args(**overrides)
    args.sglang, args.sglang_scaling = SglangConfig.parse_args(args)
    return args


def _specs(args: Namespace) -> list[InferenceEngineSpec]:
    return InferenceEngineSpec.create(args)


class TestGroupDpPortReservation:
    def test_group_dp_overrides_reserve_independent_master_port_ranges(self, tmp_path: Path) -> None:
        """Each engine group reserves ports for its resolved DP size rather than the global default."""
        config_path = tmp_path / "groups.yaml"
        config_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n"
            "        overrides:\n"
            "          dp_size: 5\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n"
        )
        args = _make_args(sglang_config=str(config_path), rollout_num_gpus=8, sglang_dp_size=3)

        specs = _specs(args)

        assert [
            next(port.num_consecutive for port in spec.port_infos if port.name == "dist_init") for spec in specs
        ] == [35, 33]


class TestEngineEnvironmentTiming:
    def test_launch_reads_current_environment_after_spec_creation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Engine relaunches observe current environment values instead of a cached specification-time snapshot."""
        monkeypatch.setenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "before")
        args = _make_args()
        (spec,) = _specs(args)
        context = Namespace(args=args)

        monkeypatch.setenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "first-launch")
        first = spec.env_var(context)
        monkeypatch.setenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "relaunch")
        second = spec.env_var(context)

        assert first["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "first-launch"
        assert second["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "relaunch"
        assert first is not second
