"""Memory profiling utilities using tracemalloc."""

import tracemalloc
from contextlib import contextmanager
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .results import Results


class MemoryProfiler:
    """Handles memory profiling using tracemalloc."""

    def __init__(self, results: "Results") -> None:
        self._results = results

    @contextmanager
    def profile(self):
        """Context manager for memory profiling.

        Yields a function to store memory peaks at labeled checkpoints.

        Example:
            with profiler.profile() as record:
                # do encoding work
                record("encoding")
                # do explanation work
                record("explanation")
        """
        try:
            tracemalloc.start()
            yield self._store_peak
        finally:
            tracemalloc.stop()

    def _store_peak(self, label: str) -> None:
        """Store current memory peak and reset."""
        _, peak = tracemalloc.get_traced_memory()
        self._results.store_custom(f"memory/{label}", peak)
        tracemalloc.reset_peak()

    def start(self) -> None:
        """Start memory tracking."""
        tracemalloc.start()

    def stop(self) -> None:
        """Stop memory tracking."""
        tracemalloc.stop()

    def get_peak(self) -> int:
        """Get peak memory usage in bytes and stop tracking."""
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak
