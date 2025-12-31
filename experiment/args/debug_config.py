"""Configuration dataclass for Experiment.debug() method."""

from dataclasses import dataclass, field
from typing import Literal, Optional


# Type aliases for valid option values
DeduplicateType = Literal["sat", "bdd", None]
EncTypeAtLeast = Literal["tot", "mtot", "kmtot", "native"]
EncTypeEq = Literal["bit", "tot", "native"]
HitmanType = Literal["sorted", "lbx", "sat"]
SolverType = Literal["gc3", "g3", "cd195", "mgh"]
ExplainAlgorithm = Literal["mus", "mcs", "both", "find_one", "var"]
SizeType = Literal["debug", "small", "custom"]


@dataclass
class DebugConfig:
    """
    Configuration for running debug experiments.

    This dataclass groups all parameters for Experiment.debug() into
    logical categories for better organization and IDE support.

    Example:
        # Use defaults
        config = DebugConfig(dataset="iris")
        ctx = Experiment.debug(config)

        # Customize encoding
        config = DebugConfig(
            dataset="mnist",
            deduplicate="bdd",
            enc_type_at_least="mtot",
        )
        ctx = Experiment.debug(config)
    """

    # Dataset configuration
    dataset: str = "iris"
    size: SizeType = "debug"

    # Encoding configuration
    deduplicate: DeduplicateType = "sat"
    enc_type_at_least: EncTypeAtLeast = "tot"
    enc_type_eq: EncTypeEq = "bit"
    ohe_dedup: bool = True

    # Solver configuration
    solver_type: SolverType = "gc3"
    h_type: HitmanType = "lbx"
    h_solver: str = "g3"

    # Explanation configuration
    explain_algorithm: ExplainAlgorithm = "both"
    proc_rounds: int = 0

    # Strategy configuration
    use_parent_strategy: bool = True
    reverse: bool = False

    @property
    def strategy(self) -> str:
        """Compute the strategy string based on flags."""
        if self.use_parent_strategy:
            return "parent"
        return "b_full" if self.reverse else "full"

    @classmethod
    def small(cls, dataset: str = "iris", **kwargs) -> "DebugConfig":
        """Create a config for small-scale debugging."""
        return cls(dataset=dataset, size="debug", **kwargs)

    @classmethod
    def full(cls, dataset: str = "iris", **kwargs) -> "DebugConfig":
        """Create a config for full-scale experiments."""
        return cls(dataset=dataset, size="small", **kwargs)

    @classmethod
    def custom(cls, dataset: str = "iris", **kwargs) -> "DebugConfig":
        """Create a config with custom model parameters."""
        return cls(dataset=dataset, size="custom", **kwargs)

    def to_exp_args(self) -> dict:
        """Convert config to experiment args dictionary."""
        return {
            "eval_freq": 1000,
            "verbose": "info",
            "size": self.size,
            "deduplicate": self.deduplicate,
            "experiment_id": 10000,
            "load_model": True,
            "output": "csv",
            "max_explain_time": 30,
            "strategy": self.strategy,
            "enc_type_at_least": self.enc_type_at_least,
            "enc_type_eq": self.enc_type_eq,
            "ohe_deduplication": self.ohe_dedup,
            "solver_type": self.solver_type,
            "explain_one": True,
            "h_solver": self.h_solver,
            "h_type": self.h_type,
            "explain_algorithm": self.explain_algorithm,
            "process_rounds": self.proc_rounds,
            "dataset": self.dataset,
        }
