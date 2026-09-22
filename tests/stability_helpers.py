"""Shared StabilityConfig for tests that need stability to fire quickly.

Most tests exercise stage transitions, snapshots or logging -- not the
stability criterion itself. They need a config that declares stability
after a handful of trials, without asserting anything about the criterion.
Keeping it in one place means a future change to the criterion touches one
file instead of a dozen.
"""

from __future__ import annotations

from dataclasses import replace

from self_fly.config.schema import ExperimentConfig, StabilityConfig


def fast_stability_config(
    window_size: int = 15, consecutive_windows_required: int = 2
) -> StabilityConfig:
    """Permissive on both components so stability fires almost at once:
    no competence requirement and an effectively unbounded policy-drift
    allowance. Never use this for tests ABOUT the criterion."""
    return StabilityConfig(
        window_size=window_size,
        min_accuracy=0.0,
        max_median_policy_delta=1.0,
        consecutive_windows_required=consecutive_windows_required,
    )


def with_fast_stability(config: ExperimentConfig, **kwargs) -> ExperimentConfig:
    return replace(config, stability=fast_stability_config(**kwargs))
