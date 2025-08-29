from typing import List, Set, Tuple
from pysat.formula import Formula, Atom, Or, And, Neg, Implies, XOr
from dd.autoref import BDD, Function
from experiment.helpers import Context


def xor(a: Function, b: Function) -> Function:
    return (a | b) & ~(a & b)


class BDDSolver:
    def __init__(self, variables: list[int], e_ctx: Context):
        self.bdd = BDD()
        self.bdd.declare(*variables)
        self.variables = [self.bdd.var(i) for i in variables]
        # self.bdd.declare(*variables)
        self.ohe = None
        self.e_ctx = e_ctx

    def is_equiv(self, a: Function, b: Function) -> bool:
        if self.ohe is None:
            return xor(a, b) == self.bdd.false
        else:
            return xor(a, b) & self.ohe == self.bdd.false

    def is_neg_equiv(self, a: Function, b: Function) -> bool:
        return self.is_equiv(~a, b)

    def dump_list(self, roots: list[Function], filename: str = "example.png"):
        self.bdd.dump(filename, roots=roots)

    def dump(self, roots: Function, filename: str = "example.png"):
        self.dump_list([roots], filename)

    def get_one_combination(self, zeros: list[Function], one: Function) -> Function:
        exp = self.bdd.true
        for i in zeros:
            exp &= ~i
        exp &= one
        return exp

    def get_one_ohe(self, input_set: list[Function]) -> Function:
        exp = self.bdd.false
        for idx in range(len(input_set)):
            exp |= self.get_one_combination(
                input_set[:idx] + input_set[idx + 1 :], input_set[idx]
            )
        return exp

    def set_ohe(self, attribute_ranges):
        # TODO: set OHE
        start = 0
        input_sets = []
        for step in attribute_ranges:
            input_sets.append(self.variables[start : start + step])
            start += step

        exp = self.bdd.true
        for i_set in input_sets:
            exp &= self.get_one_ohe(i_set)

        self.ohe = exp

        return self

    def transform(self, formula: Formula) -> Function:
        if isinstance(formula, Atom):
            if formula == Atom(False):
                return self.bdd.false

            elif formula == Atom(True):
                return self.bdd.true

            return self.bdd.var(formula.object)

        elif isinstance(formula, Or):
            exp = self.bdd.false
            for i in formula.subformulas:
                exp |= self.transform(i)
            return exp

        elif isinstance(formula, And):
            exp = self.bdd.true
            for i in formula.subformulas:
                exp &= self.transform(i)
            return exp

        elif isinstance(formula, Neg):
            return ~self.transform(formula.subformula)

        elif isinstance(formula, Implies):
            return self.transform(formula.left).implies(self.transform(formula.right))

        elif isinstance(formula, XOr):
            v = self.transform(formula.subformulas[0])
            for i in range(1, len(formula.subformulas)):
                v = xor(v, self.transform(formula.subformulas[i]))
            return v
        else:
            raise ValueError(f"Unsupported formula type: {type(formula)}")

    def deduplicate(
        self, f: Formula, previous: Set[Tuple[int, Formula]], layer: int
    ):  # layer is 1-based
        if f == Atom(True) or f == Atom(False):
            return f

        transformed = self.transform(f)
        if self.is_equiv(transformed, self.bdd.true):
            self.e_ctx.inc_deduplication(layer, -1)
            return Atom(True)

        if self.is_equiv(transformed, self.bdd.false):
            self.e_ctx.inc_deduplication(layer, -1)
            return Atom(False)

        for p_layer, p in previous:
            # if len(str(f)) <= len(str(p)):
            #     continue
            if self.is_equiv(transformed, self.transform(p)):
                self.e_ctx.inc_deduplication(layer, p_layer)
                return p
            elif self.is_neg_equiv(transformed, self.transform(p)):
                self.e_ctx.inc_deduplication(layer, p_layer)
                return Neg(p)
        return f

    @staticmethod
    def from_inputs(inputs: List[Atom], e_ctx: Context):
        return BDDSolver([i.object for i in inputs], e_ctx=e_ctx)

    def __del__(self):
        del self.bdd
