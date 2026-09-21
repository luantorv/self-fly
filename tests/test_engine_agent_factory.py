from dataclasses import replace

import pytest

from self_fly.config.defaults import default_config
from self_fly.experiment.engine import ExperimentEngine


def test_invalid_agent_type_fails_loud():
    config = replace(default_config(seed=0), agent_type="not_a_real_agent")
    with pytest.raises(ValueError):
        ExperimentEngine(config)
