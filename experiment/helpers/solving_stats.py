"""SAT solving statistics tracking."""

from typing import Optional


class SolvingStats:
    """Tracks SAT solver clause and variable statistics."""

    def __init__(self) -> None:
        self._num_clauses: int = 0
        self._num_vars: int = 0
        self._solving_num_clauses: list[int] = []
        self._solving_num_vars: list[int] = []

    @property
    def num_clauses(self) -> int:
        """Total number of clauses in the encoding."""
        return self._num_clauses

    @property
    def num_vars(self) -> int:
        """Total number of variables in the encoding."""
        return self._num_vars

    def store_encoding_stats(self, clauses: list[list[int]]) -> None:
        """Store clause and variable counts from the encoding.

        Args:
            clauses: List of CNF clauses
        """
        self._num_clauses = len(clauses)
        if clauses:
            self._num_vars = max(
                abs(literal) for clause in clauses for literal in clause
            )
        else:
            self._num_vars = 0

    def record_solving_instance(self, num_clauses: int, num_vars: int) -> None:
        """Record statistics for a single solving instance.

        Args:
            num_clauses: Number of clauses used in this solve
            num_vars: Number of variables used in this solve
        """
        self._solving_num_clauses.append(num_clauses)
        self._solving_num_vars.append(num_vars)

    def get_avg_clauses(self) -> float:
        """Get average clauses per solving instance."""
        if not self._solving_num_clauses:
            return 0.0
        return sum(self._solving_num_clauses) / len(self._solving_num_clauses)

    def get_avg_vars(self) -> float:
        """Get average variables per solving instance."""
        if not self._solving_num_vars:
            return 0.0
        return sum(self._solving_num_vars) / len(self._solving_num_vars)
