import logging
import torch

from pysat.formula import Formula, Atom, CNF, Or, And
from pysat.card import CardEnc, EncType

from difflogic import LogicLayer, GroupSum

from experiment.helpers import Context, SatContext
from lgn.dataset import AutoTransformer
from lgn.encoding import Encoding
from lgn.encoding.encoders.util import get_eq_constraints
from lgn.encoding.util import get_parts


logger = logging.getLogger(__name__)


class Encoder:
    def __init__(self, e_ctx: Context):
        self.e_ctx = e_ctx

    def get_encoding(
        self,
        model,
        Dataset: AutoTransformer,
    ):
        self.context = SatContext()

        formula, input_handles = self.get_formula(
            model, Dataset.get_input_dim(), Dataset
        )
        input_ids, cnf, output_ids, special = self.populate_clauses(
            input_handles=input_handles, formula=formula
        )

        # REMARK: formula represents output from second last layer
        # ie.: dimension is neuron_number, not class number

        eq_constraints = self.initialize_ohe(
            Dataset, input_ids, self.e_ctx.get_enc_type()
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

        return x, inputs

    def __handle_output_duplicates(self, output_ids, cnf):
        total_output_len = len(output_ids)
        cls_no = self.e_ctx.dataset.get_num_of_classes()
        assert total_output_len % cls_no == 0
        group_size = total_output_len // cls_no

        duplicates = dict()  # key: output var, value: list of indices in output_ids

        for idx in range(0, total_output_len, group_size):
            subgroup = output_ids[idx : idx + group_size]
            rest = output_ids[:idx]
            # rest = output_ids[:idx] + output_ids[idx + group_size :] Only compare with previous, previous takes precendence
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

    def populate_clauses(self, input_handles, formula):
        with self.use_context() as vpool:
            input_ids = [vpool.id(h) for h in input_handles]
            cnf = CNF()
            output_ids = []
            special = dict()
            # adding the clauses to a global CNF
            idx = 0

            for f, g in zip(
                [And(Atom(True), f.simplified()) for f in formula], formula
            ):
                # print("f", f)
                # print("g", g)
                # print("list(g)", list(g))
                # print("f", f.simplified())
                # print("list(f)", list(f))
                l = list(f)
                if len(l) == 0:
                    special[idx] = f.simplified()
                    output_ids.extend([None])
                else:
                    # f.clausify()
                    # print("list(f)[:-1]", list(f)[:-1])
                    # print("'f.clauses[-1][1]", f.clauses[-1][1])
                    # print("list(f)[-1]", list(f)[-1])
                    cnf.extend(list(f)[:-1])
                    # logger.debug("Formula: %s", f)
                    # logger.debug("CNF Clauses: %s", f.clauses)
                    # logger.debug("Simplified: %s", f.simplified())
                    # logger.debug("CNF Clauses: %s", cnf.clauses)
                    # print("(Populate Clauses) vpool", vpool.id2obj.items())
                    # input()

                    # if f.clauses[-1][1] is None:
                    # special[idx] = f.simplified()
                    output_ids.extend(list(f)[-1])
                # input("Press Enter to Continue...")
                idx += 1

                # logger.debug("=== === === ===")
            logger.debug("CNF Clauses: %s", cnf.clauses)

            logger.debug("output_ids: %s", str(output_ids))

        output_ids, cnf = self.__handle_output_duplicates(output_ids, cnf)

        return input_ids, cnf, output_ids, special

    def initialize_ohe(self, Dataset: AutoTransformer, input_ids, enc_type):
        with self.use_context() as vpool:
            return get_eq_constraints(Dataset, input_ids, enc_type, vpool)

    def use_context(self):
        return self.context.use_vpool()
