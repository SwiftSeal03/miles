import pytest
from pydantic import ValidationError

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.component_trainer import TrainerOnlyConfig
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace, compute_custom_function_config
from miles.utils.args.schema import BaseConfig


class _HookConfig(BaseConfig):
    temperature: float


class TestCustomFunctionView:
    def test_nested_hooks_and_runtime_inputs_are_visible_without_mutating_sources(self) -> None:
        """A nested callback sees its parent, nested hook and runtime values together."""
        args = ImmutableNamespace(model="model", temperature=0.5)
        hook = CustomFunctionConfig(path="outer", config=_HookConfig(temperature=0.5))
        nested = CustomFunctionConfig(path="inner", config=ImmutableNamespace(limit=7))

        view = compute_custom_function_config(args, hook, {"step": 3}, nested_functions=[None, nested])

        assert dict(view) == {"model": "model", "temperature": 0.5, "limit": 7, "step": 3}
        assert dict(args) == {"model": "model", "temperature": 0.5}
        with pytest.raises(ValidationError, match="frozen"):
            view.step = 4

    @pytest.mark.parametrize("source", ["hook", "nested", "runtime"])
    def test_conflicting_sources_are_rejected_instead_of_silently_overriding(self, source: str) -> None:
        """No hook source may silently override a field owned by another configuration."""
        args = ImmutableNamespace(temperature=0.5)
        conflicting = CustomFunctionConfig(path="conflict", config=_HookConfig(temperature=0.9))
        hook = conflicting if source == "hook" else CustomFunctionConfig(path="outer")
        nested = [conflicting] if source == "nested" else []
        runtime = {"temperature": 0.9} if source == "runtime" else {}

        with pytest.raises(ValueError, match="temperature.*conflicting values"):
            compute_custom_function_config(args, hook, runtime, nested_functions=nested)

    def test_trainer_backend_fields_reach_legacy_hooks(self) -> None:
        """Legacy trainer hooks see backend-owned values in their flat configuration."""
        args = TrainerOnlyConfig(
            backend=MegatronArgsNamespace(tensor_model_parallel_size=4),
            trainer_role="actor",
            trainer_actor_index=2,
        )

        view = compute_custom_function_config(args, CustomFunctionConfig(path="legacy"))

        assert view.tensor_model_parallel_size == 4
        assert view.trainer_role == "actor"
        assert view.trainer_actor_index == 2

    def test_equal_duplicates_are_accepted_across_all_sources(self) -> None:
        """Shared hook settings may appear repeatedly when their values agree."""
        args = ImmutableNamespace(temperature=0.5)
        hook = CustomFunctionConfig(path="outer", config=_HookConfig(temperature=0.5))

        view = compute_custom_function_config(args, hook, {"temperature": 0.5}, nested_functions=[hook])

        assert view.temperature == 0.5
