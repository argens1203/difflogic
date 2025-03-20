from pysat.solvers import Solver as BaseSolver


class Solver:
    def __init__(self):
        self.solver = BaseSolver()

    def append_formula(self, clauses=[]):
        self.solver.append_formula(clauses)
        return self

    def delete(self):
        self.solver.delete()

    def solve(self, assumptions=[]):
        return self.solver.solve(assumptions=assumptions)
