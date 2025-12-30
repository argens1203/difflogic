"""Experiment results storage and serialization."""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from lgn.encoding import Encoding


class Results:
    """Stores and serializes experiment results.

    Handles timing, metrics, and JSON serialization for experiment tracking.

    Attributes:
        eid: Experiment identifier
        path: Directory path for saving results
        test_acc: Test accuracy (set via store_test_acc)
    """

    def __init__(self, eid: int, path: str) -> None:
        """Initialize results storage.

        Args:
            eid: Unique experiment identifier
            path: Directory path for saving JSON results
        """
        self.eid: int = eid
        self.path: str = path

        self.init_time: float = time.time()
        self.save_time: Optional[float] = None
        self.total_time: Optional[float] = None

        self.args: Optional[dict[str, Any]] = None
        self.server_name: str = socket.gethostname().split(".")[0]

        # Timing attributes (set by store_* methods)
        self.start_time: float = 0.0
        self.model_ready_time: float = 0.0
        self.encoding_ready_time: float = 0.0
        self.explanation_ready_time: float = 0.0
        self.end_time: float = 0.0

        # Metrics (set during experiment)
        self.test_acc: float = 0.0

    def store_args(self, args: Any) -> None:
        """Store experiment arguments.

        Args:
            args: Arguments object (will be converted via vars())
        """
        self.args = vars(args)

    def store_results(self, results: dict[str, Any]) -> None:
        """Append results to lists by key.

        Args:
            results: Dictionary of key-value pairs to append
        """
        for key, val in results.items():
            if not hasattr(self, key):
                setattr(self, key, list())
            getattr(self, key).append(val)

    def store_final_results(self, results: dict[str, Any]) -> None:
        """Store final results with underscore suffix.

        Args:
            results: Dictionary of final key-value pairs
        """
        for key, val in results.items():
            setattr(self, key + "_", val)

    def save(self) -> None:
        """Save results to JSON file."""
        self.save_time = time.time()
        self.total_time = self.save_time - self.init_time

        json_str = json.dumps(self.__dict__, indent=4, sort_keys=True)

        filepath = os.path.join(self.path, "{:08d}.json".format(self.eid))
        with open(filepath, mode="w") as f:
            f.write(json_str)

    @staticmethod
    def load(
        eid: int, path: str, get_dict: bool = False
    ) -> Union["Results", dict[str, Any]]:
        """Load results from JSON file.

        Args:
            eid: Experiment identifier
            path: Directory path containing results
            get_dict: If True, return raw dictionary instead of Results

        Returns:
            Results object or dictionary depending on get_dict
        """
        filepath = os.path.join(path, "{:08d}.json".format(eid))
        with open(filepath, mode="r") as f:
            data = json.loads(f.read())

        if get_dict:
            return data

        result = Results(-1, "")
        result.__dict__.update(data)

        assert eid == result.eid
        return result

    # === Encoding Storage ===

    def store_encoding(self, encoding: "Encoding") -> None:
        """Store encoding statistics.

        Args:
            encoding: The encoding object with stats
        """
        self.cnf_size: int = encoding.get_stats()["clauses_size"]
        self.eq_size: int = encoding.get_stats()["eq_size"]
        self.formulas: list[str] = [str(f.simplified()) for f in encoding.formula]
        self.encoding_time: float = time.time()
        self.encoding_time_taken: float = self.encoding_time - self.model_ready_time

    # === Explanation Statistics ===

    def store_explanation_stat(
        self, mean_explain_count: float, deduplication: int
    ) -> None:
        """Store explanation statistics.

        Args:
            mean_explain_count: Average explanation count
            deduplication: Deduplication count
        """
        self.mean_explain_count: float = mean_explain_count
        self.deduplication: int = deduplication

    def store_resource_usage(
        self, mean_explain_time: float, memory_usage: int
    ) -> None:
        """Store resource usage metrics.

        Args:
            mean_explain_time: Average time per explanation
            memory_usage: Memory usage in bytes
        """
        self.mean_explain_time: float = mean_explain_time
        self.memory_usage: int = memory_usage

    def store_counts(self, instance_count: int, explain_count: int) -> None:
        """Store instance and explanation counts.

        Args:
            instance_count: Number of instances processed
            explain_count: Number of explanations generated
        """
        self.instance_count: int = instance_count
        self.explanation_count: int = explain_count

    def store_custom(self, key: str, val: Any) -> None:
        """Store a custom key-value pair.

        Args:
            key: Attribute name
            val: Value to store
        """
        setattr(self, key, val)

    def store_test_acc(self, test_acc: float) -> None:
        """Store test accuracy.

        Args:
            test_acc: Test accuracy value
        """
        self.test_acc = test_acc

    # === Timing Methods ===

    def store_start_time(self) -> None:
        """Record experiment start time."""
        self.start_time = time.time()

    def store_model_ready_time(self) -> None:
        """Record when model is ready."""
        self.model_ready_time = time.time()

    def store_encoding_ready_time(self) -> None:
        """Record when encoding is ready."""
        self.encoding_ready_time = time.time()

    def store_explanation_ready_time(self) -> None:
        """Record when explanations are ready."""
        self.explanation_ready_time = time.time()

    def store_end_time(self) -> None:
        """Record experiment end time."""
        self.end_time = time.time()

    def get_model_ready_time(self) -> float:
        """Get time taken for model preparation.

        Returns:
            Time in seconds from start to model ready
        """
        return self.model_ready_time - self.start_time

    def get_encoding_time(self) -> float:
        """Get time taken for encoding.

        Returns:
            Time in seconds from model ready to encoding ready
        """
        return self.encoding_ready_time - self.model_ready_time

    def get_explanation_time(self) -> float:
        """Get time taken for explanations.

        Returns:
            Time in seconds from encoding ready to explanations ready
        """
        return self.explanation_ready_time - self.encoding_ready_time

    def get_total_runtime(self) -> float:
        """Get total experiment runtime.

        Returns:
            Time in seconds from start to end
        """
        return self.end_time - self.start_time

    def get_value(self, key: str) -> Optional[Any]:
        """Get a stored value by key.

        Args:
            key: Attribute name

        Returns:
            Stored value or None if not found
        """
        return getattr(self, key, None)


if __name__ == "__main__":
    r = Results(101, "./")
    print(r.__dict__)
    r.save()

    r2 = Results.load(101, "./")
    print(r2.__dict__)
