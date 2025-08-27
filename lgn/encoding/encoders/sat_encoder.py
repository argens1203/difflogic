import logging

from pysat.formula import Atom, Neg

from experiment.helpers import SatContext

from lgn.dataset import AutoTransformer

from .sat_deduplicator import DeduplicationMixin
from .encoder import Encoder
from ..encoding import Encoding
from .util import _get_layers, get_eq_constraints


logger = logging.getLogger(__name__)


class SatEncoder(Encoder):
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

    def d(self, input_handles, model, const_lookup, is_rev_lookup, pair_lookup):
        curr = input_handles
        lookup = dict()
        for i, layer in enumerate(_get_layers(model)):
            curr = layer.get_formula(curr)
            special = {}
            for j, g in enumerate(curr):
                if (i, j) in const_lookup:
                    lookup[(i, j)] = Atom(const_lookup[(i, j)])
                    curr[j] = lookup[(i, j)]
                    special[j] = lookup[(i, j)]
                    if g != Atom(True) and g != Atom(False):
                        self.e_ctx.inc_deduplication()
                elif (i, j) in is_rev_lookup:
                    if is_rev_lookup[(i, j)]:
                        lookup[(i, j)] = Neg(lookup[pair_lookup[(i, j)]])
                        curr[j] = lookup[(i, j)]
                    else:
                        lookup[(i, j)] = lookup[pair_lookup[(i, j)]]
                        curr[j] = lookup[(i, j)]
                    self.e_ctx.inc_deduplication()
                else:
                    lookup[(i, j)] = g
        return curr, special

    def get_encoding(self, model, Dataset: AutoTransformer):
        gates, const_lookup, is_rev_lookup, pair_lookup = DeduplicationMixin(
            self.e_ctx
        )._function(model, Dataset)

        self.context = SatContext()
        input_handles, input_ids = self._get_inputs(Dataset)

        formula, special = self.d(
            input_handles, model, const_lookup, is_rev_lookup, pair_lookup
        )
        output_ids = gates[-1]
        #

        # with self.use_context() as vpool:
        # print("next vpool var", vpool._next())
        # input("Press enter to continue...")

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
