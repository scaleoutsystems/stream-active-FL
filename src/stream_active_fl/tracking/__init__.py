"""
Run tracking and metrics persistence.

- Run setup: git info, environment info, run_info.json (`run`)
- StreamingMetricsLogger: CSV logging for buffer-based streaming training
- FederatedMetricsLogger: round-based CSV logging for federated training
- FederatedDecisionsLogger: per-frame decision log for federated training
"""

from stream_active_fl.runtime import create_run_dir

from .federated import FederatedDecisionsLogger, FederatedMetricsLogger
from .run import (
    get_environment_info,
    get_git_info,
    log_gpu_memory,
    save_run_info,
)
from .streaming import StreamingMetricsLogger

__all__ = [
    "FederatedDecisionsLogger",
    "FederatedMetricsLogger",
    "StreamingMetricsLogger",
    "create_run_dir",
    "get_environment_info",
    "get_git_info",
    "log_gpu_memory",
    "save_run_info",
]
