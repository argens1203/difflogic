"""Deduplication tracking for encoding optimization."""

from collections import OrderedDict
from typing import Union


# Layer target constants
CONSTANT_LAYER = "Constant"
INPUT_LAYER = "Input"

LayerTarget = Union[int, str]


class DeduplicationTracker:
    """Tracks deduplication statistics during encoding."""

    def __init__(self) -> None:
        self._count: int = 0
        self._layer_seen: set[int] = set()
        self._dedup_dict: OrderedDict[tuple[int, LayerTarget], int] = OrderedDict()
        self._ohe_deduplication: list[tuple[int, int]] = []

    @property
    def count(self) -> int:
        """Total deduplication count."""
        return self._count

    @property
    def ohe_deduplication(self) -> list[tuple[int, int]]:
        """List of OHE deduplication pairs."""
        return self._ohe_deduplication

    @property
    def dedup_dict(self) -> OrderedDict[tuple[int, LayerTarget], int]:
        """Deduplication counts by (layer, target) pair."""
        return self._dedup_dict

    def reset(self) -> None:
        """Reset all deduplication tracking."""
        self._count = 0
        self._dedup_dict = OrderedDict()

    def increment(self, curr_layer: int, target_layer: int) -> None:
        """Record a deduplication event.

        Args:
            curr_layer: Current layer (1-based)
            target_layer: Target layer (1-based, 0=input, -1=constants)
        """
        if curr_layer not in self._layer_seen:
            self._layer_seen.add(curr_layer)
            self._dedup_dict[(curr_layer, CONSTANT_LAYER)] = 0
            self._dedup_dict[(curr_layer, INPUT_LAYER)] = 0
            for k in range(1, curr_layer + 1):
                self._dedup_dict[(curr_layer, k)] = 0

        # Convert special layer indices to strings
        target: LayerTarget
        if target_layer == -1:
            target = CONSTANT_LAYER
        elif target_layer == 0:
            target = INPUT_LAYER
        else:
            target = target_layer

        self._count += 1
        self._dedup_dict[(curr_layer, target)] = (
            self._dedup_dict.get((curr_layer, target), 0) + 1
        )

    def increment_ohe(self, ohe_from: int, ohe_to: int) -> None:
        """Record an OHE deduplication event."""
        self._ohe_deduplication.append((ohe_from, ohe_to))

    def print_summary(self) -> None:
        """Print deduplication summary by layer."""
        for (layer, target), count in self._dedup_dict.items():
            print(f"Layer {layer} -> {target}: {count}")
