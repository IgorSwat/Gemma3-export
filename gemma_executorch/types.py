from typing import Any, Optional, Sequence
import dataclasses
import enum

"""
  Type definitions for Gemma models
"""

class Architecture(enum.Enum):
    GEMMA_1 = 1
    GEMMA_2 = 2
    GEMMA_3 = 3


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2

