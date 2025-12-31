"""Experiment context for managing state during runs."""

from datetime import datetime
import csv
import logging
from typing import Any, Callable

from tabulate import tabulate
import torch

from .logging import setup_logger
from .util import seed_all, get_results, get_enc_type
from .results import Results
from .memory_profiler import MemoryProfiler
from .deduplication_tracker import DeduplicationTracker
from .solving_stats import SolvingStats
from .output_formatter import OutputFormatter
from lgn.dataset import load_dataset


class CacheKey:
    """Cache key constants."""

    SOLVER = "solver"


# Backward compatibility alias
Cached_Key = CacheKey


class Context:
    """Experiment context managing state, logging, and metrics.

    This class coordinates various trackers and provides a unified interface
    for experiment execution.

    Attributes:
        args: Experiment configuration
        results: Results storage
        dataset: Loaded dataset
        train_loader: Training data loader
        test_loader: Test data loader
        dedup_tracker: Deduplication statistics tracker
        solving_stats: SAT solving statistics tracker
        memory_profiler: Memory profiling utility
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        setup_logger(args)
        seed_all(args.seed)

        self.logger = logging.getLogger(__name__)

        eid = args.experiment_id if args.experiment_id is not None else 0
        self.results: Results = get_results(eid, args)
        self.train_loader, self.test_loader, self.get_raw, self.dataset = load_dataset(
            args
        )
        self.verbose = args.verbose
        self.enc_type_at_least = get_enc_type(args.enc_type_at_least)
        self.enc_type_eq = get_enc_type(args.enc_type_eq)
        self.solver_type = args.solver_type
        self.fp_type = torch.float32

        # Cache tracking
        self.cache_hit = {CacheKey.SOLVER: 0}
        self.cache_miss = {CacheKey.SOLVER: 0}

        # Initialize focused trackers
        self.dedup_tracker = DeduplicationTracker()
        self.solving_stats = SolvingStats()
        self.memory_profiler = MemoryProfiler(self.results)
        self._output_formatter = OutputFormatter(self)

        self.results.store_start_time()
        self.num_explanations = 0

    # === Logging ===

    def debug(self, fn: Callable[[], Any]) -> None:
        """Execute function only in debug mode."""
        if self.verbose == "debug":
            fn()

    # === Memory Profiling (delegated) ===

    def use_memory_profile(self):
        """Context manager for memory profiling."""
        return self.memory_profiler.profile()

    def store_memory_peak(self, label: str = None) -> None:
        """Store current memory peak."""
        self.memory_profiler._store_peak(label)

    def start_memory_usage(self) -> None:
        """Start memory tracking."""
        self.memory_profiler.start()

    def end_memory_usage(self) -> None:
        """Stop memory tracking."""
        self.memory_profiler.stop()

    def get_memory_usage(self, label: str = None) -> int:
        """Get peak memory usage."""
        return self.memory_profiler.get_peak()

    # === Cache Tracking ===

    def inc_cache_hit(self, flag: str) -> None:
        """Increment cache hit counter."""
        self.cache_hit[flag] += 1

    def inc_cache_miss(self, flag: str) -> None:
        """Increment cache miss counter."""
        self.cache_miss[flag] += 1

    # === Deduplication (delegated, with backward compatibility) ===

    @property
    def deduplication(self) -> int:
        """Total deduplication count."""
        return self.dedup_tracker.count

    @property
    def ohe_deduplication(self) -> list[tuple[int, int]]:
        """OHE deduplication pairs."""
        return self.dedup_tracker.ohe_deduplication

    @property
    def dedup_dict(self):
        """Deduplication dictionary."""
        return self.dedup_tracker.dedup_dict

    @property
    def layer_seen(self) -> set[int]:
        """Layers that have been seen."""
        return self.dedup_tracker._layer_seen

    def reset_deduplication(self) -> None:
        """Reset deduplication tracking."""
        self.dedup_tracker.reset()

    def inc_deduplication(self, curr_layer: int, target_layer: int) -> None:
        """Increment deduplication counter."""
        self.dedup_tracker.increment(curr_layer, target_layer)

    def inc_ohe_deduplication(self, ohe_from: int, ohe_to: int) -> None:
        """Increment OHE deduplication counter."""
        self.dedup_tracker.increment_ohe(ohe_from, ohe_to)

    def print_dedup_dict(self) -> None:
        """Print deduplication summary."""
        self.dedup_tracker.print_summary()

    # === Solving Stats (delegated, with backward compatibility) ===

    @property
    def num_clauses(self) -> int:
        """Number of clauses."""
        return self.solving_stats.num_clauses

    @property
    def num_vars(self) -> int:
        """Number of variables."""
        return self.solving_stats.num_vars

    @property
    def solving_num_clauses(self) -> list[int]:
        """Per-instance clause counts."""
        return self.solving_stats._solving_num_clauses

    @property
    def solving_num_vars(self) -> list[int]:
        """Per-instance variable counts."""
        return self.solving_stats._solving_num_vars

    def store_clause(self, clauses: list[list[int]]) -> None:
        """Store clause statistics."""
        self.solving_stats.store_encoding_stats(clauses)

    def record_solving_stats(self, num_clauses: int, num_vars: int) -> None:
        """Record solving instance statistics."""
        self.solving_stats.record_solving_instance(num_clauses, num_vars)

    def get_avg_solving_clauses(self) -> float:
        """Get average clauses per solve."""
        return self.solving_stats.get_avg_clauses()

    def get_avg_solving_vars(self) -> float:
        """Get average variables per solve."""
        return self.solving_stats.get_avg_vars()

    # === Explanation Counting ===

    def inc_num_explanations(self, num: int) -> None:
        """Increment explanation count."""
        self.num_explanations += num

    # === Output (delegated) ===

    def output(self) -> None:
        """Output results based on args.output format."""
        if self.args.output == "display":
            self.display()
        elif self.args.output == "csv":
            self.to_csv()
        else:
            raise ValueError(f"Unknown output format: {self.args.output}")

    def get_headers(self) -> list[str]:
        """Get output headers."""
        return self._output_formatter.get_headers()

    def get_data(self) -> list[list[Any]]:
        """Get formatted output data."""
        return self._output_formatter.get_data()

    def to_csv(self) -> None:
        """Write results to CSV."""
        self._output_formatter.to_csv()

    def display(self) -> None:
        """Display results as table."""
        self._output_formatter.display()

    # === Config Accessors ===

    def get_enc_type(self):
        """Get encoding type for at-least constraints."""
        return self.enc_type_at_least

    def get_enc_type_eq(self):
        """Get encoding type for equality constraints."""
        return self.enc_type_eq

    def get_solver_type(self) -> str:
        """Get solver type."""
        return self.solver_type

    def get_fp_type(self):
        """Get floating point type."""
        return self.fp_type

    def get_dataset(self):
        """Get the loaded dataset."""
        return self.dataset

    def get_process_rounds(self) -> int:
        """Get number of process rounds."""
        return self.args.process_rounds

    def __del__(self) -> None:
        """Log cache statistics on destruction."""
        self.logger.debug("Cache Hit: %s", str(self.cache_hit))
        self.logger.debug("Cache Miss: %s", str(self.cache_miss))
        self.logger.debug("Deduplication: %s", str(self.deduplication))


class MultiContext:
    """Aggregates results from multiple experiment runs."""

    def __init__(self) -> None:
        self.data: list[list[Any]] = []
        self.headers: list[str] = None
        self.dedup_dicts: list = []

    def add(self, ctx: Context) -> None:
        """Add a context's results to the aggregation."""
        self.data.append(ctx.get_data()[0])
        self.dedup_dicts.append(ctx.dedup_dict)
        self.headers = ctx.get_headers()

    @staticmethod
    def _unique_timestamp() -> str:
        """Generate unique timestamp string."""
        return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

    def to_csv(self, filename: str, with_timestamp: bool = True) -> None:
        """Write aggregated results to CSV."""
        if with_timestamp:
            filename = f"{self._unique_timestamp()}_{filename}"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.data)

    def display(self) -> None:
        """Display aggregated results as table."""
        print(tabulate(self.data, headers=self.headers, tablefmt="github"))

    # Backward compatibility
    @property
    def dedup_dict(self):
        """Alias for dedup_dicts."""
        return self.dedup_dicts
