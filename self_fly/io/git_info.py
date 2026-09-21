from __future__ import annotations

import subprocess


def current_git_revision() -> str | None:
    """The checked-out commit hash, or None outside a git checkout (e.g. a
    packaged install) -- reproducibility claims should degrade gracefully,
    not crash the run."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
