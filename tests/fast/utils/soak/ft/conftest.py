from datetime import datetime, timedelta, timezone

import pytest
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTargetConfig
from tests.utils.soak.core.events import SoakAdmissionClosedEvent, SoakEvent, SoakRunContext, SoakRunContextEvent


@pytest.fixture
def events() -> list[SoakEvent]:
    now = datetime.now(timezone.utc) - timedelta(seconds=1)
    return [
        SoakRunContextEvent(
            timestamp=now,
            context=SoakRunContext(
                base_url="http://localhost:18080",
                config=SoakRunnerConfig(
                    seed=42,
                    target_configs={"rollout": SoakTargetConfig(expected_count=1, mean_interval_seconds=120)},
                ),
                form_names={},
                train_config=None,
            ),
            sources={},
        ),
        SoakAdmissionClosedEvent(timestamp=now),
    ]

