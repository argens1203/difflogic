"""Base encoder for converting neural network models to SAT formulas."""

import logging
from typing import Any, Optional

import torch
from pysat.formula import Formula, Atom, CNF, Or, And, IDPool
from pysat.card import CardEnc, EncType

from difflogic import LogicLayer, GroupSum
from experiment.helpers import Context, SatContext
from lgn.dataset import AutoTransformer
from lgn.encoding import Encoding
from lgn.encoding.encoders.util import get_eq_constraints
from lgn.encoding.util import get_parts


logger = logging.getLogger(__name__)


class Encoder:
    """Encodes a logic gate neural network as a SAT formula.

    Converts LogicLayer and GroupSum layers into CNF clauses that can be
    used for formal verification and explanation generation.

    Attributes:
        e_ctx: Experiment context for configuration and tracking
        context: SAT context for variable pool management
    """

    def __init__(self, e_ctx: Context) -> None:
        """Initialize the encoder.

        Args:
            e_ctx: Experiment context with configuration
        """
        self.e_ctx = e_ctx
        self.context: Optional[SatContext] = None

    def get_encoding(
        self,
        model: Any,
        Dataset: AutoTransformer,
    ) -> Encoding:
        """Generate a complete encoding of the model.

        Args:
            model: Neural network model with LogicLayer/GroupSum layers
            Dataset: Dataset transformer with input/output specifications

        Returns:
            Encoding object containing CNF clauses and metadata
        """
        self.context = SatContext()

        formula, input_handles = self.get_formula(
            model, Dataset.get_input_dim(), Dataset
        )
        logger.debug('formula: "%s"', formula)

        input_ids, cnf, output_ids, special = self.populate_clauses(
            input_handles=input_handles, formula=formula
        )

        # REMARK: formula represents output from second last layer
        # ie.: dimension is neuron_number, not class number

        eq_constraints = self.initialize_ohe(
            Dataset, input_ids, self.e_ctx.get_enc_type_eq()
        )

        return Encoding(
            clauses=cnf.clauses,
            eq_constraints=eq_constraints,
            input_ids=input_ids,
            output_ids=output_ids,
            formula=formula,
            special=special,
            s_ctx=self.context,
            e_ctx=self.e_ctx,
        )

    def get_formula(
        self,
        model: Any,
        input_dim: int,
        Dataset: AutoTransformer,
    ) -> tuple[list[Formula], list[Atom]]:
        """Convert model layers to Boolean formulas.

        Args:
            model: Neural network model
            input_dim: Number of input dimensions
            Dataset: Dataset transformer

        Returns:
            Tuple of (output formulas, input atoms)
        """
        with self.use_context() as vpool:
            x: list[Atom] = [Atom(i + 1) for i in range(input_dim)]
            inputs = x

            logger.debug("Not deduplicating...")

            for layer in model:
                assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
                if isinstance(layer, GroupSum):
                    continue
                x = layer.get_formula(x)

        return x, inputs

    def __handle_output_duplicates(
        self,
        output_ids: list[Optional[int]],
        cnf: CNF,
    ) -> tuple[list[Optional[int]], CNF]:
        """Handle duplicate output variables across classes.

        Args:
            output_ids: List of output variable IDs
            cnf: CNF formula to extend

        Returns:
            Tuple of (updated output_ids, updated CNF)
        """
        total_output_len = len(output_ids)
        cls_no = self.e_ctx.dataset.get_num_of_classes()
        assert total_output_len % cls_no == 0
        group_size = total_output_len // cls_no

        duplicates: dict[int, list[int]] = {}

        for idx in range(0, total_output_len, group_size):
            subgroup = output_ids[idx : idx + group_size]
            rest = output_ids[:idx]
            for o in subgroup:
                if o is not None and -o in rest:
                    duplicates.setdefault(o, []).append(idx)

        logger.debug("Before deduplication output_ids: %s", str(output_ids))

        with self.use_context() as vpool:
            for k, v in duplicates.items():
                for idx in v:
                    auxvar = vpool._next()
                    cnf.extend([[k, -auxvar], [-k, auxvar]])
                    output_ids[idx] = auxvar

        logger.debug("Final output_ids: %s", str(output_ids))

        return output_ids, cnf

    def populate_clauses(
        self,
        input_handles: list[Atom],
        formula: list[Formula],
    ) -> tuple[list[int], CNF, list[Optional[int]], dict[int, Atom]]:
        """Convert formulas to CNF clauses.

        Args:
            input_handles: Input atom handles
            formula: List of output formulas

        Returns:
            Tuple of (input_ids, CNF, output_ids, special constants)
        """
        with self.use_context() as vpool:
            input_ids: list[int] = [vpool.id(h) for h in input_handles]
            cnf = CNF()
            output_ids: list[Optional[int]] = []
            special: dict[int, Atom] = {}
            idx = 0

            for f, g in zip(
                [And(Atom(True), f.simplified()) for f in formula], formula
            ):
                clause_list = list(f)
                if len(clause_list) == 0:
                    special[idx] = f.simplified()
                    output_ids.extend([None])
                else:
                    cnf.extend(list(f)[:-1])
                    output_ids.extend(list(f)[-1])
                idx += 1

            logger.debug("CNF Clauses: %s", cnf.clauses)
            logger.debug("output_ids: %s", str(output_ids))

        return input_ids, cnf, output_ids, special

    def initialize_ohe(
        self,
        Dataset: AutoTransformer,
        input_ids: list[int],
        enc_type: int,
    ) -> CNF:
        """Initialize one-hot encoding constraints.

        Args:
            Dataset: Dataset transformer with attribute ranges
            input_ids: Input variable IDs
            enc_type: Encoding type for cardinality constraints

        Returns:
            CNF with equality constraints
        """
        with self.use_context() as vpool:
            return get_eq_constraints(Dataset, input_ids, enc_type, vpool)

    def use_context(self) -> Any:
        """Get the variable pool context manager.

        Returns:
            Context manager for the SAT variable pool
        """
        return self.context.use_vpool()
