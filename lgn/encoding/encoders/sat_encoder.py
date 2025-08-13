from tqdm import tqdm
import logging
import torch

from pysat.formula import Atom, Neg, Implies, Or

from difflogic import LogicLayer, GroupSum

from constant import Stats
from pysat.solvers import Solver as BaseSolver

from experiment.helpers.ordered_set import OrderedSet
from lgn.dataset import AutoTransformer
from .sat_deduplicator import DeduplicationMixin
from .encoder import Encoder
from experiment.helpers import SatContext
from ..encoding import Encoding

from pysat.formula import Formula
from constant import Stats
from pysat.card import EncType

logger = logging.getLogger(__name__)

fp_type = torch.float32


class SatEncoder(Encoder, DeduplicationMixin):
    def get_encoding(self, model, Dataset: AutoTransformer, fp_type=fp_type, **kwargs):

        self.context = SatContext()

        clauses = []
        Stats["deduplication"] = 0

        with self.use_context() as vpool:
            print("vpool, sat_encoder", id(vpool))
            #  GET input handles
            input_handles = [Atom(i + 1) for i in range(Dataset.get_input_dim())]
            all = OrderedSet()
            for i in input_handles:
                all.add(i)
            print("all", all)
            input("Press Enter to continue...")
            x = input_handles

            # Get solver
            input_ids = [vpool.id(h) for h in input_handles]
            eq_constraints, parts = self.initialize_ohe(
                Dataset, input_ids, enc_type=kwargs.get("enc_type", EncType.totalizer)
            )
            print("eq_constraints", eq_constraints)
            print("parts", parts)
            input("Press Enter to continue...")
            self.solver = BaseSolver(name="g3")
            self.solver.append_formula(eq_constraints)  # OHE
            # # res = self.deduplicate_pair(Or(Atom(3), Neg(Atom(4))), Atom(3))
            # res = self.deduplicate_pair(Neg(Implies(Atom(3), Atom(4))), Atom(3))

            # print("res", res)
            # assert (
            #     res is True
            # ), "Deduplication failed for Neg(Implies(Atom(3), Atom(4)))"
            # input("Press N-TERRRR to continue...")

            for layer in model:
                this_layer = []
                assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
                if isinstance(layer, GroupSum):
                    continue
                print("before deduplication")
                layer.print()
                for f in tqdm(layer.get_formula(x)):
                    f = self.deduplicate(f, all)
                    # print("(Populate Clauses) vpool", vpool.id2obj.items())
                    # input()
                    f.clausify()
                    this_layer.append(f)
                    all.add(f)
                    clauses.extend(f.clauses)
                x = this_layer
                print("after deduplication")
                print(x)
                input("Press Enter to continue...")

            clauses = [y.clauses for y in x]
            print("clauses", clauses)

            input_ids, cnf, output_ids, special = self.populate_clauses(
                input_handles=input_handles, formula=x
            )

            return Encoding(
                parts=parts,
                cnf=cnf,
                eq_constraints=eq_constraints,
                fp_type=fp_type,
                Dataset=Dataset,
                input_ids=input_ids,
                output_ids=output_ids,
                formula=x,
                input_handles=input_handles,
                special=special,
                enc_type=kwargs.get("enc_type", EncType.totalizer),
                context=self.context,
            )
        # return Encoding.from_clauses(clauses)

    # def deduplicate(self, formula, all) -> Formula:
    #     pass

    # first_encoding = Encoder().get_encoding(
    #     model, Dataset, fp_type=fp_type, **kwargs
    # )
    # deduplicator = SatDeduplicator(first_encoding)

    # enc_type = kwargs.get("enc_type", EncType.totalizer)
    # self.context = Context()

    # input_dim = Dataset.get_input_dim()
    # class_dim = Dataset.get_num_of_classes()

    # formula, input_handles = self.get_formula(
    #     model, input_dim, deduplicator=deduplicator
    # )
    # input_ids, cnf, output_ids, special = self.populate_clauses(
    #     input_handles=input_handles, formula=formula
    # )

    # eq_constraints, parts = self.initialize_ohe(Dataset, input_ids, enc_type)
    # deduplicator.delete()

    # return Encoding(
    #     parts=parts,
    #     cnf=cnf,
    #     eq_constraints=eq_constraints,
    #     input_dim=input_dim,
    #     fp_type=fp_type,
    #     Dataset=Dataset,
    #     class_dim=class_dim,
    #     input_ids=input_ids,
    #     output_ids=output_ids,
    #     formula=formula,
    #     input_handles=input_handles,
    #     special=special,
    #     enc_type=enc_type,
    #     context=self.context,
    # )

    # def get_formula(
    #     self,
    #     model,
    #     input_dim,
    #     deduplicator: SatDeduplicator,
    # ):
    #     with self.use_context() as vpool:
    #         x = [Atom(i + 1) for i in range(input_dim)]
    #         inputs = x

    #         logger.debug("Deduplicating with SAT solver ...")
    #         all = set()
    #         for i in x:
    #             all.add(i)
    #         Stats["deduplication"] = 0

    #         for layer in model:
    #             assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
    #             if isinstance(layer, GroupSum):
    #                 continue
    #             x = layer.get_formula(x)
    #             assert x is not None, "Layer returned None"
    #             for idx in tqdm(range(len(x))):
    #                 x[idx] = deduplicator.deduplicate(x[idx], all)
    #                 all.add(x[idx])
    #                 assert x[idx] is not None, "Deduplicator returned None"

    #                 print("(Sat Encoder) vpool", vpool.id2obj.items())
    #                 input()

    #     return x, inputs
