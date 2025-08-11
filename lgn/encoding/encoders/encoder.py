import logging
import torch


from pysat.formula import Formula, Atom, CNF, Or
from pysat.card import CardEnc, EncType

from difflogic import LogicLayer, GroupSum


from lgn.dataset import AutoTransformer
from lgn.encoding import Encoding
from ..context import Context

fp_type = torch.float32

logger = logging.getLogger(__name__)


class Encoder:
    def get_encoding(
        self,
        model,
        Dataset: AutoTransformer,
        fp_type=fp_type,
        **kwargs,
    ):
        enc_type = kwargs.get("enc_type", EncType.totalizer)
        self.context = Context()

        formula, input_handles = self.get_formula(
            model, Dataset.get_input_dim(), Dataset
        )
        input_ids, cnf, output_ids, special = self.populate_clauses(
            input_handles=input_handles, formula=formula
        )

        # REMARK: formula represents output from second last layer
        # ie.: dimension is neuron_number, not class number

        eq_constraints, parts = self.initialize_ohe(Dataset, input_ids, enc_type)

        print("formula", formula)
        print("input_handles", input_handles)
        print("input_ids", input_ids)
        print("cnf", cnf)
        print("output_ids", output_ids)
        print("special", special)
        print("eq_constraints", eq_constraints)
        print("parts", parts)
        input("Press Enter to continue...")

        return Encoding(
            parts=parts,
            cnf=cnf,
            eq_constraints=eq_constraints,
            fp_type=fp_type,
            Dataset=Dataset,
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
        # TODO: second return is actually list[Atom] but cannot be defined as such
    ) -> tuple[list[Formula], list[Formula]]:
        with self.use_context() as vpool:
            x = [Atom(i + 1) for i in range(input_dim)]
            inputs = x

            logger.debug("Not deduplicating...")

            for layer in model:
                assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
                if isinstance(layer, GroupSum):  # TODO: make get_formula for GroupSum
                    continue
                x = layer.get_formula(x)

                print("(Encoder): vpool", vpool.id2obj.items())

        return x, inputs

    def populate_clauses(self, input_handles, formula):
        print("formula.length", len(formula))
        with self.use_context() as vpool:
            input_ids = [vpool.id(h) for h in input_handles]
            cnf = CNF()
            output_ids = []
            special = dict()
            # adding the clauses to a global CNF
            idx = 0
            for f in [Or(Atom(False), f.simplified()) for f in formula]:
                f.clausify()
                cnf.extend(list(f)[:-1])
                logger.debug("Formula: %s", f)
                logger.debug("CNF Clauses: %s", f.clauses)
                logger.debug("Simplified: %s", f.simplified())
                logger.debug("CNF Clauses: %s", cnf.clauses)
                # print("(Populate Clauses) vpool", vpool.id2obj.items())
                # input()
                if f.clauses[-1][1] is None:
                    special[idx] = f.simplified()
                idx += 1
                output_ids.append(f.clauses[-1][1])

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
