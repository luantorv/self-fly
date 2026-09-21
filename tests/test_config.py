import json

from self_fly.config.defaults import default_config
from self_fly.config.loader import config_from_dict, config_to_dict, load_config, save_config


def test_json_roundtrip_dict():
    cfg = default_config(seed=42, run_name="test_run")
    data = config_to_dict(cfg)
    cfg2 = config_from_dict(data)
    assert cfg2 == cfg


def test_save_load_file(tmp_path):
    cfg = default_config(seed=7)
    path = tmp_path / "config.json"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg
    json.loads(path.read_text())  # valid JSON text


def test_stage_reward_overrides_present():
    cfg = default_config()
    stage1 = next(s for s in cfg.stages if s.stage == 1)
    assert stage1.reward_overrides["self_a|REAL"] == 0.0
    assert stage1.reward_overrides["self_a|FALSA"] == 0.0
    assert stage1.reward_overrides["self_a|NO_SE"] == 0.0
