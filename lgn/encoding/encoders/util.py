"""Utility functions for SAT encoding."""

import logging
from typing import Any

from pysat.formula import CNF, IDPool
from pysat.card import CardEnc

from difflogic import LogicLayer, GroupSum
from lgn.dataset import AutoTransformer
from lgn.encoding.util import get_parts


logger = logging.getLogger(__name__)


def _get_layers(model: Any) -> list[LogicLayer]:
    """Extract LogicLayer instances from a model.

    Filters out GroupSum layers and returns only LogicLayer instances.

    Args:
        model: Iterable model containing LogicLayer and/or GroupSum layers

    Returns:
        List of LogicLayer instances
    """
    layers: list[LogicLayer] = []
    for layer in model:
        assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
        if isinstance(layer, GroupSum):
            continue
        layers.append(layer)
    return layers


def get_eq_constraints(
    Dataset: AutoTransformer,
    input_ids: list[int],
    enc_type: int,
    vpool: IDPool,
) -> CNF:
    """Generate equality constraints for one-hot encoded inputs.

    Creates cardinality constraints ensuring exactly one variable is true
    within each one-hot encoded attribute group.

    Args:
        Dataset: Dataset with attribute range information
        input_ids: List of variable IDs for inputs
        enc_type: PySAT encoding type for cardinality constraints
        vpool: Variable pool for auxiliary variables

    Returns:
        CNF containing equality constraints
    """
    eq_constraints = CNF()
    parts = get_parts(Dataset, input_ids)

    logger.debug("full_input_ids: %s", input_ids)
    logger.debug("parts: %s", parts)

    for part in parts:
        eq_constraints.extend(
            CardEnc.equals(lits=part, vpool=vpool, encoding=enc_type)
        )

    logger.debug("eq_constraints: %s", eq_constraints.clauses)

    return eq_constraints
