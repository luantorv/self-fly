from __future__ import annotations

from enum import Enum


class Action(Enum):
    REAL = "REAL"
    FALSA = "FALSA"
    NO_SE = "NO_SE"


ACTIONS = (Action.REAL, Action.FALSA, Action.NO_SE)
