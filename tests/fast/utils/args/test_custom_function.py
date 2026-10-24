import argparse
import pickle
from types import SimpleNamespace
from typing import Annotated
import pytest
from miles.utils.args.custom_function import CustomFunctionConfig, _adapt_legacy_custom_config, resolve_custom_function_configs
from miles.utils.args.schema import Arg, BaseConfig
from miles.utils.function_registry import function_registry

class _HookConfig(BaseConfig):
    threshold: Annotated[float, Arg()] = 0.5

class _RunConfig(BaseConfig):
    custom_megatron_init_path: CustomFunctionConfig
    custom_megatron_before_log_prob_hook_path: CustomFunctionConfig | None

class TestCustomFunctionResolution:

    def test_shared_hook_fields_are_removed_only_after_both_hooks_are_built(self) -> None:
        """Multiple hooks sharing a field must each receive it before the flat field is removed."""
        args = argparse.Namespace(custom_megatron_init_path='hook', custom_megatron_before_log_prob_hook_path='hook', threshold=0.8, unrelated=7)
        with function_registry.temporary('hook', SimpleNamespace(config_class=_HookConfig)):
            resolve_custom_function_configs(_fill_hook_defaults(args))
        assert args.custom_megatron_init_path.config.threshold == 0.8
        assert args.custom_megatron_before_log_prob_hook_path.config.threshold == 0.8
        assert vars(args).keys() == _hook_field_names() | {'unrelated'}
        assert args.unrelated == 7

    def test_an_absent_optional_hook_remains_absent(self) -> None:
        """Optional hooks must not be replaced with empty path configurations."""
        args = argparse.Namespace(custom_megatron_init_path='hook', custom_megatron_before_log_prob_hook_path=None, threshold=0.8)
        with function_registry.temporary('hook', SimpleNamespace(config_class=_HookConfig)):
            resolve_custom_function_configs(_fill_hook_defaults(args))
        assert args.custom_megatron_before_log_prob_hook_path is None
        assert args.custom_megatron_init_path.path == 'hook'

    def test_typed_hook_configuration_survives_json_and_pickle(self) -> None:
        """Transport resolves the hook's concrete schema instead of losing custom fields."""
        config = CustomFunctionConfig(path='hook', config=_HookConfig(threshold=0.7))
        with function_registry.temporary('hook', SimpleNamespace(config_class=_HookConfig)):
            restored = CustomFunctionConfig.model_validate_json(config.model_dump_json())
            unpickled = pickle.loads(pickle.dumps(config))
        assert type(restored.config) is _HookConfig
        assert restored.config.threshold == 0.7
        assert unpickled == restored

    def test_serialized_hooks_cannot_fill_missing_fields_from_defaults(self) -> None:
        """Incomplete hook payloads are rejected even when local schema defaults exist."""
        with function_registry.temporary('hook', SimpleNamespace(config_class=_HookConfig)):
            with pytest.raises(ValueError, match='Incomplete configuration.*threshold'):
                CustomFunctionConfig.model_validate({'path': 'hook', 'config': {}})

class TestLegacyCustomArguments:

    def test_positional_callbacks_and_required_fields_keep_their_contract(self) -> None:
        """Legacy positional callbacks retain required fields and CLI value conversion."""

        def register(parser: argparse.ArgumentParser, /) -> argparse.ArgumentParser:
            parser.add_argument('--required-value', required=True, type=int)
            parser.add_argument('--optional-value', default='fallback')
            return parser
        config_class = _adapt_legacy_custom_config(register)
        parser = argparse.ArgumentParser()
        config_class.add_arguments(parser=parser)
        parsed = parser.parse_args(['--required-value', '9'])
        assert config_class.model_validate(vars(parsed)).model_dump() == {'required_value': 9, 'optional_value': 'fallback'}
        with pytest.raises(ValueError):
            config_class.model_validate({})

    def test_suppressed_defaults_cannot_be_silently_synthesized(self) -> None:
        """A legacy suppressed argument remains required in the transport schema."""

        def register(parser: argparse.ArgumentParser) -> None:
            parser.add_argument('--value', default=argparse.SUPPRESS)
        config_class = _adapt_legacy_custom_config(register)
        with pytest.raises(ValueError):
            config_class.model_validate({})
        assert config_class.model_validate({'value': 'present'}).model_dump() == {'value': 'present'}

class TestCustomArgumentRegistration:

    def test_shared_config_classes_are_registered_once_across_distinct_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Distinct hooks sharing a schema must not register duplicate option strings."""
        from miles.utils.args.custom_function import add_user_provided_function_arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--custom-megatron-init-path', default='first-hook')
        parser.add_argument('--custom-megatron-before-log-prob-hook-path', default='second-hook')
        monkeypatch.setattr('sys.argv', ['test', '--threshold', '0.9'])
        with function_registry.temporary('first-hook', SimpleNamespace(config_class=_HookConfig)):
            with function_registry.temporary('second-hook', SimpleNamespace(config_class=_HookConfig)):
                result = add_user_provided_function_arguments(parser, modify_args=_fill_hook_defaults)
        assert result is parser
        assert parser.parse_args(['--threshold', '0.9']).threshold == 0.9

    def test_partial_discovery_defers_missing_modules_until_final_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Partial parsing tolerates unavailable hooks while final resolution fails explicitly."""
        from miles.utils.args.custom_function import add_user_provided_function_arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--custom-megatron-init-path', default='missing_test_hook_package.hook')
        parser.add_argument('--custom-megatron-before-log-prob-hook-path', default=None)
        monkeypatch.setattr('sys.argv', ['test'])
        assert add_user_provided_function_arguments(parser, modify_args=_fill_hook_defaults) is parser
        with pytest.raises(ModuleNotFoundError):
            resolve_custom_function_configs(_fill_hook_defaults(parser.parse_args([])))
from miles.utils.args.runtime import AllConfig

def _hook_field_names() -> set[str]:
    return {name for name, field in AllConfig.model_fields.items() if field.annotation in {CustomFunctionConfig, CustomFunctionConfig | None}}

def _fill_hook_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for name in _hook_field_names():
        vars(args).setdefault(name, None)
    return args
