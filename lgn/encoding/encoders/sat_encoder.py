from tqdm import tqdm
import logging
import torch

from difflogic import LogicLayer, GroupSum

from pysat.formula import Atom
from pysat.card import EncType
from pysat.solvers import Solver as BaseSolver

from constant import Stats

from experiment.helpers.ordered_set import OrderedSet
from experiment.helpers import SatContext

from lgn.dataset import AutoTransformer

from .sat_deduplicator import DeduplicationMixin
from .encoder import Encoder
from ..encoding import Encoding


logger = logging.getLogger(__name__)

fp_type = torch.float32


class SatEncoder(Encoder, DeduplicationMixin):
    def get_encoding(self, model, Dataset: AutoTransformer, fp_type=fp_type, **kwargs):

        self.context = SatContext()

        clauses = []
        Stats["deduplication"] = 0

        with self.use_context() as vpool:
            #  GET input handles
            input_handles = [Atom(i + 1) for i in range(Dataset.get_input_dim())]
            all = OrderedSet()
            for i in input_handles:
                all.add(i)
            x = input_handles

            # Get solver
            input_ids = [vpool.id(h) for h in input_handles]
            eq_constraints, parts = self.initialize_ohe(
                Dataset, input_ids, enc_type=kwargs.get("enc_type", EncType.totalizer)
            )
            self.solver = BaseSolver(name="g3")
            self.solver.append_formula(eq_constraints)  # OHE

            for layer in model:
                this_layer = []
                layer.print()
                assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
                if isinstance(layer, GroupSum):
                    continue
                for f in tqdm(layer.get_formula(x)):
                    f = self.deduplicate(f, all)
                    f.clausify()
                    this_layer.append(f)
                    if f != Atom(True) and f != Atom(False):
                        all.add(f)
                    clauses.extend(f.clauses)
                x = this_layer

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
