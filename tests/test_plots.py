from self_fly.config.defaults import default_config
from self_fly.experiment.engine import ExperimentEngine
from self_fly.io.run_manager import RunManager
from self_fly.visualization.plots import generate_all_plots, load_trials_jsonl


def test_generate_all_plots_from_saved_trials_headless(tmp_path):
    config = default_config(seed=0, run_name="plot_test")
    run_manager = RunManager(config, base_dir=tmp_path)
    engine = ExperimentEngine(config, logger=run_manager.logger)
    engine.run(300)

    trials = load_trials_jsonl(run_manager.logger.trials_path)
    assert len(trials) == 300

    output_dir = run_manager.run_dir / "plots"
    generate_all_plots(run_manager.logger.trials_path, output_dir)

    action_probs_png = output_dir / "action_probabilities.png"
    reward_png = output_dir / "reward.png"
    assert action_probs_png.exists() and action_probs_png.stat().st_size > 0
    assert reward_png.exists() and reward_png.stat().st_size > 0
