import logging
from contextlib import contextmanager

from pysat.examples.hitman import Hitman

from .multiclass_solver import MulticlassSolver
from .instance import Instance
from lgn.util import Inp, Partial_Inp, Htype, Partial_Inp_Set

logger = logging.getLogger(__name__)

from typing import List, Iterator


class Session:
    def __init__(self, instance: Instance, hitman: Hitman, oracle: MulticlassSolver):
        self.instance = instance
        self.hitman = hitman
        self.duals = []
        self.expls = []
        self.oracle = oracle
        self.pred_class = instance.get_predicted_class()
        self.itr = 0
        pass

    def is_solvable_with(self, inp: Partial_Inp_Set):
        return self.oracle.is_solvable(pred_class=self.pred_class, inp=list(inp))

    def solve(self, inp: Partial_Inp_Set):
        return self.oracle.solve(
            pred_class=self.pred_class,
            inp=list(inp),
        )

    def hit(self, hypo: Partial_Inp):
        self.hitman.hit(hypo)
        self.duals.append(hypo)

    def block(self, hypo: Partial_Inp):
        self.hitman.block(hypo)
        self.expls.append(hypo)
        pass

    def get(self):
        self.itr += 1
        hset = self.hitman.get()
        logger.debug("itr %s) cand: %s", self.itr, hset)
        return hset

    def add_to_itr(self, value: int):
        self.itr += value

    def get_itr(self):
        return self.itr

    def get_duals(self):
        return self.duals

    def get_expls(self):
        return self.expls

    def get_expls_count(self):
        return len(self.expls)

    @contextmanager
    def use_context(
        instance: Instance, hit_type: Htype = "lbx", oracle=None
    ) -> Iterator["Session"]:
        try:
            hitman = Hitman(
                bootstrap_with=[instance.get_input()],
                htype=hit_type,
            )
            yield Session(instance, hitman=hitman, oracle=oracle)
        finally:
            hitman.delete()  # Cleanup
