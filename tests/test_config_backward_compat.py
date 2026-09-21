from self_fly.config.loader import config_from_dict
from self_fly.config.defaults import default_config
from self_fly.config.loader import config_to_dict


def test_pre_phase_b_config_dict_still_loads():
    """A config.json saved by Phase A never had agent_type/experimental_condition
    -- config_from_dict must fill in the ExperimentConfig defaults, not crash."""
    phase_a_data = config_to_dict(default_config(seed=3, run_name="phase_a_run"))
    del phase_a_data["agent_type"]
    del phase_a_data["experimental_condition"]

    loaded = config_from_dict(phase_a_data)

    assert loaded.agent_type == "baseline"
    assert loaded.experimental_condition == "baseline"
    assert loaded.seed == 3
