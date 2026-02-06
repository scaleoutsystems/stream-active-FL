"""
Memory management for continual learning.

Bounded memory structures for experience replay and rehearsal:
- ReplayBuffer: Bounded buffer with configurable admission policies (FIFO, random, reservoir)
"""

from .replay_buffer import ReplayBuffer

__all__ = ["ReplayBuffer"]
