from difflogic import LogicLayer, GroupSum
from pysat.formula import CNF
from pysat.card import CardEnc

from difflogic import LogicLayer, GroupSum

import logging
from lgn.dataset import AutoTransformer
from lgn.encoding.util import get_parts

logger = logging.getLogger(__name__)


def _get_layers(model) -> list[LogicLayer]:
    layers = []
    for layer in model:
        assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
        if isinstance(layer, GroupSum):
            continue
        layers.append(layer)
    return layers


def get_eq_constraints(Dataset: AutoTransformer, input_ids, enc_type, vpool):
    eq_constraints = CNF()
    parts = get_parts(Dataset, input_ids)

    logger.debug("full_input_ids: %s", input_ids)

    for part in parts:
        eq_constraints.extend(
            CardEnc.equals(
                lits=part,
                vpool=vpool,
                encoding=enc_type,
            )
        )

    logger.debug("eq_constraints: %s", eq_constraints.clauses)

    return eq_constraints
