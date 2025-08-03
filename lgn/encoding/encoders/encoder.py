import logging
import torch


from pysat.formula import Formula, Atom, CNF, Or
from pysat.card import CardEnc, EncType

from difflogic import LogicLayer, GroupSum


from lgn.dataset import AutoTransformer
from ..deduplicator import SatDeduplicator
from lgn.encoding import Encoding
from ..context import Context

fp_type = torch.float32

logger = logging.getLogger(__name__)


class Encoder:
    def get_static(
        self,
        model,
        Dataset: AutoTransformer,
        fp_type=fp_type,
        **kwargs,
    ):
        enc_type = kwargs.get("enc_type", EncType.totalizer)
        self.context = Context()

        input_dim = Dataset.get_input_dim()
        class_dim = Dataset.get_num_of_classes()

        deduplicator = kwargs.get("deduplicator", None)

        formula, input_handles = self.get_formula(
            model, input_dim, Dataset, deduplicator=deduplicator
        )
        input_ids, cnf, output_ids, special = self.populate_clauses(
            input_handles=input_handles, formula=formula
        )

        # REMARK: formula represents output from second last layer
        # ie.: dimension is neuron_number, not class number

        eq_constraints, parts = self.initialize_ohe(Dataset, input_ids, enc_type)

        return Encoding(
            parts=parts,
            cnf=cnf,
            eq_constraints=eq_constraints,
            input_dim=input_dim,
            fp_type=fp_type,
            Dataset=Dataset,
            class_dim=class_dim,
            input_ids=input_ids,
            output_ids=output_ids,
            formula=formula,
            input_handles=input_handles,
            special=special,
            enc_type=enc_type,
            context=self.context,
        )

    def get_formula(
        self,
        model,
        input_dim,
        Dataset: AutoTransformer,
        deduplicator: SatDeduplicator,
        # TODO: second return is actually list[Atom] but cannot be defined as such
    ) -> tuple[list[Formula], list[Formula]]:
        with self.use_context():
            x = [Atom(i + 1) for i in range(input_dim)]
            inputs = x

            logger.debug("Not deduplicating...")

            for layer in model:
                assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
                if isinstance(layer, GroupSum):  # TODO: make get_formula for GroupSum
                    continue
                x = layer.get_formula(x)

        return x, inputs

    def populate_clauses(self, input_handles, formula):
        with self.use_context() as vpool:
            input_ids = [vpool.id(h) for h in input_handles]
            cnf = CNF()
            output_ids = []
            special = dict()
            # adding the clauses to a global CNF
            for f in [Or(Atom(False), f.simplified()) for f in formula]:
                f.clausify()
                cnf.extend(list(f)[:-1])
                logger.debug("Formula: %s", f)
                logger.debug("CNF Clauses: %s", f.clauses)
                logger.debug("Simplified: %s", f.simplified())
                logger.debug("CNF Clauses: %s", cnf.clauses)
                idx = 0
                if f.clauses[-1][1] is None:
                    special[idx] = f.simplified()
                output_ids.append(f.clauses[-1][1])
                idx += 1

                logger.debug("=== === === ===")
            logger.debug("CNF Clauses: %s", cnf.clauses)

        return input_ids, cnf, output_ids, special

    def initialize_ohe(self, Dataset: AutoTransformer, input_ids, enc_type):
        eq_constraints = CNF()
        parts: list[list[int]] = []
        with self.use_context() as vpool:
            start = 0
            logger.debug("full_input_ids: %s", input_ids)
            for step in Dataset.get_attribute_ranges():
                logger.debug("Step: %d", step)
                logger.debug("input_ids: %s", input_ids[start : start + step])
                part = input_ids[start : start + step]
                eq_constraints.extend(
                    CardEnc.equals(
                        lits=part,
                        vpool=vpool,
                        encoding=enc_type,
                    )
                )
                start += step
                parts.append(part)
        logger.debug("eq_constraints: %s", eq_constraints.clauses)

        return eq_constraints, parts

    def use_context(self):
        return self.context.use_vpool()
