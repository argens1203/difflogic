import logging
from contextlib import contextmanager

from pysat.examples.hitman import Hitman

from .multiclass_solver import MulticlassSolver
from .instance import Instance
from lgn.util import (
    Inp,
    Partial_Inp,
    Htype,
    Partial_Inp_Set,
    Transformed_Partial_Inp_Set,
)

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

        # NEW
        self.inp2raw = dict()
        for i, inp in enumerate(instance.get_input_as_set()):
            self.inp2raw[i + 1] = inp

        self.raw2inp = dict()
        for i, set_ in self.inp2raw.items():
            for raw in set_:
                self.raw2inp[raw] = i

        self.raw2inp_new = dict()
        for i, set_ in self.inp2raw.items():
            self.raw2inp_new[frozenset(set_)] = i

        self.options = list(self.inp2raw.keys())
        # NEW
        pass

        # NEW -- BEGIN

    def transformed_set_to_raw(self, inp: Transformed_Partial_Inp_Set):
        actual_inp = set()
        for i in inp:
            actual_inp = actual_inp.union(self.inp2raw[i])
        return actual_inp

    def is_solvable_with_opt(self, inp: Transformed_Partial_Inp_Set):
        return self.oracle.is_solvable(
            pred_class=self.pred_class,
            inp=self.transformed_set_to_raw(inp),
        )

    def translate(self, actual_inp: Partial_Inp_Set):
        overlap = actual_inp & self.instance.get_input()
        ret = set()
        for set_, translation in self.raw2inp_new.items():
            print(translation, set_)
            if set_ - overlap == set():
                ret.add(translation)
            else:
                ret.add(-translation)
        return ret

    def solve_opt(self, inp: Transformed_Partial_Inp_Set):
        res = self.solve(
            inp=self.transformed_set_to_raw(inp),
        )
        if res["model"] is not None:
            res["model"] = self.translate(res["model"])
        if res["core"] is not None:
            res["core"] = self.translate(res["core"])
        return res

    def get_duals_opt(self):
        return [list(self.transformed_set_to_raw(dual)) for dual in self.duals]

    def get_expls_opt(self):
        return [(list(self.transformed_set_to_raw(expl))) for expl in self.expls]
        # NEW

    def is_solvable_with(self, inp: Partial_Inp_Set):
        return self.oracle.is_solvable(pred_class=self.pred_class, inp=list(inp))

    def solve(self, inp: Partial_Inp_Set):
        # NEW
        print()
        print("solve - inp: ", inp)
        # NEW
        res = self.oracle.solve(
            pred_class=self.pred_class,
            inp=list(inp),
        )
        if res["model"] is not None:
            res["model"] = set(res["model"])

        if res["core"] is not None:
            res["core"] = set(res["core"])
        return res

    def hit(self, hypo: Partial_Inp_Set):
        self.hitman.hit(list(hypo))
        self.duals.append(hypo)

    def block(self, hypo: Partial_Inp_Set):
        self.hitman.block(list(hypo))
        self.expls.append(hypo)
        pass

    def get(self):
        self.itr += 1
        hset = self.hitman.get()
        logger.debug("itr %s) cand: %s", self.itr, hset)

        if hset is None:
            return None

        return set(hset)

    def add_to_itr(self, value: int):
        self.itr += value

    def get_itr(self):
        return self.itr

    def get_duals(self):
        return [list(dual) for dual in self.duals]

    def get_expls(self):
        return [list(expl) for expl in self.expls]

    def get_expls_count(self):
        return len(self.expls)

    @contextmanager
    def use_context(
        instance: Instance, hit_type: Htype = "lbx", oracle=None
    ) -> Iterator["Session"]:
        try:
            bs = list(range(1, len(instance.grouped_inp) + 1))
            hitman = Hitman(
                bootstrap_with=[bs],
                htype=hit_type,
            )
            yield Session(instance, hitman=hitman, oracle=oracle)
        finally:
            hitman.delete()  # Cleanup
