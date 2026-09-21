from __future__ import annotations

import statistics


def classify_dt_series(dt_values: list[float | None], window: int = 20) -> dict:
    """Purely descriptive characterization of a D_t = JS(pi_t, pi_{t-1})
    series: window-mean envelope, a count of abrupt-change spikes (values
    far from the median in units of MAD), a coarse linear drift slope over
    that envelope, and the tail's coefficient of variation (oscillating vs.
    settling). These are unvalidated heuristics for telling convergence,
    oscillation, drift, and abrupt change apart at a glance -- never
    collapse them into a single binary 'converged' verdict."""
    values = [v for v in dt_values if v is not None]
    if not values:
        return {"n": 0}

    n = len(values)
    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values]) or 1e-9
    abrupt_change_count = sum(1 for v in values if abs(v - median) > 5 * mad)

    windows = [values[i : i + window] for i in range(0, n, window)]
    window_means = [sum(w) / len(w) for w in windows if w]

    if len(window_means) >= 2:
        xs = list(range(len(window_means)))
        mean_x = sum(xs) / len(xs)
        mean_y = sum(window_means) / len(window_means)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, window_means))
        denominator = sum((x - mean_x) ** 2 for x in xs) or 1e-9
        drift_slope = numerator / denominator
    else:
        drift_slope = 0.0

    tail = values[-window:]
    tail_mean = sum(tail) / len(tail)
    tail_std = statistics.pstdev(tail) if len(tail) > 1 else 0.0
    tail_cv = (tail_std / tail_mean) if tail_mean != 0 else float("inf")

    return {
        "n": n,
        "median": median,
        "mad": mad,
        "abrupt_change_count": abrupt_change_count,
        "window_means": window_means,
        "drift_slope": drift_slope,
        "tail_mean": tail_mean,
        "tail_cv": tail_cv,
    }
