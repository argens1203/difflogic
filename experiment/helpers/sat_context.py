"""SAT formula context management for variable pool isolation."""

from contextlib import contextmanager
from typing import Any, Generator

from pysat.formula import Formula, IDPool


class SatContext:
    """Manages isolated variable pool context for SAT formula construction.

    Each SatContext maintains a separate variable pool, allowing multiple
    encodings to be constructed independently without variable ID conflicts.

    Example:
        ctx = SatContext()
        with ctx.use_vpool() as vpool:
            # All Formula operations use this context's vpool
            atom = Formula.atom("x")
    """

    def __init__(self) -> None:
        """Initialize a new SAT context with a unique identifier."""
        self.vpool_context: int = id(self)

    def get_vpool_context(self) -> int:
        """Get the variable pool context identifier.

        Returns:
            Unique context identifier
        """
        return self.vpool_context

    @contextmanager
    def use_vpool(self) -> Generator[IDPool, None, None]:
        """Context manager for using this context's variable pool.

        Temporarily sets the global Formula context to this context's pool,
        restoring the previous context on exit.

        Yields:
            The active IDPool for variable management

        Example:
            with ctx.use_vpool() as vpool:
                var_id = vpool.id("my_variable")
        """
        prev: int = Formula._context
        try:
            Formula.set_context(self.vpool_context)
            yield Formula.export_vpool(active=True)
        finally:
            Formula.set_context(prev)

    def __del__(self) -> None:
        """Clean up the variable pool on deletion."""
        self.delete()

    def delete(self) -> None:
        """Explicitly clean up the variable pool context."""
        Formula.cleanup(self.vpool_context)
