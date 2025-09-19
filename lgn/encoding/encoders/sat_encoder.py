import logging

from pysat.formula import Atom, Neg

from experiment.helpers import SatContext

from lgn.dataset import AutoTransformer
from lgn.encoding.encoders.new_sat_deduplicator import NewSatDeduplicator

from .sat_deduplicator import SatDeduplicator
from .encoder import Encoder
from ..encoding import Encoding
from .util import _get_layers, get_eq_constraints


logger = logging.getLogger(__name__)


class SatEncoder(Encoder):
    strategy = "full"

    def _get_inputs(self, Dataset: AutoTransformer):
        with self.use_context() as vpool:
            input_handles = [Atom(i + 1) for i in range(Dataset.get_input_dim())]
            input_ids = [vpool.id(h) for h in input_handles]
        return input_handles, input_ids

    def _get_eq_constraints(self, input_ids):
        with self.use_context() as vpool:
            return get_eq_constraints(
                self.e_ctx.get_dataset(),
                input_ids,
                enc_type=self.e_ctx.get_enc_type(),
                vpool=vpool,
            )

    def _get_formula(
        self, input_handles, model, const_lookup, is_rev_lookup, pair_lookup
    ):
        curr = input_handles
        lookup = dict()
        for i, h in enumerate(input_handles):
            lookup[(0, i)] = h

        i = 1  # First layer is inputs
        for layer in _get_layers(model):
            curr = layer.get_formula(curr)
            special = {}
            for j, g in enumerate(curr):
                if (i, j) in const_lookup:
                    f = Atom(const_lookup[(i, j)])
                    lookup[(i, j)] = f
                    curr[j] = f
                    special[j] = f
                    if g != Atom(True) and g != Atom(False):
                        self.e_ctx.inc_deduplication(i, -1)
                elif (i, j) in is_rev_lookup:
                    target = pair_lookup[(i, j)]
                    if is_rev_lookup[(i, j)]:
                        lookup[(i, j)] = Neg(lookup[target])
                        curr[j] = lookup[(i, j)]
                    else:
                        lookup[(i, j)] = lookup[target]
                        curr[j] = lookup[(i, j)]
                    self.e_ctx.inc_deduplication(i, target[0])
                else:
                    lookup[(i, j)] = g
            i += 1
        return curr, special

    def get_encoding(self, model, Dataset: AutoTransformer):
        if self.strategy in ["full", "b_full", "parent"]:
            const_lookup, is_rev_lookup, pair_lookup = SatDeduplicator(
                self.e_ctx
            ).deduplicate(model, Dataset, strategy=self.strategy)
        else:
            const_lookup, is_rev_lookup, pair_lookup = SatDeduplicator(
                self.e_ctx
            ).deduplicate(model, Dataset, strategy=self.strategy)

        self.context = SatContext()

        input_handles, input_ids = self._get_inputs(Dataset)
        formula, special = self._get_formula(
            input_handles, model, const_lookup, is_rev_lookup, pair_lookup
        )

        input_ids, cnf, output_ids, special = self.populate_clauses(
            input_handles, formula
        )

        return Encoding(
            clauses=cnf.clauses,
            eq_constraints=self._get_eq_constraints(input_ids),
            input_ids=input_ids,
            output_ids=output_ids,
            formula=formula,
            special=special,
            s_ctx=self.context,
            e_ctx=self.e_ctx,
        )
