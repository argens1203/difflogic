import torch
from constant import device
from .util import formula_as_pseudo_model, get_truth_table_loader
from pysat.formula import Formula, Atom, IDPool
from difflogic import LogicLayer, GroupSum
from contextlib import contextmanager

fp_type = torch.float32


def get_formula(model, input_dim):
    # x = [Atom() for i in range(input_dim)]
    x = [Atom(i + 1) for i in range(input_dim)]
    inputs = x
    all = set()
    for i in x:
        all.add(i)

    for layer in model:
        assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
        if isinstance(layer, GroupSum):  # TODO: make get_formula for GroupSum
            continue
        x = layer.get_formula(x)
        for o in x:
            all.add(o)

    return x, inputs


class PseudoModel:
    def __init__(self, model, input_dim, output_dim, fp_type=fp_type):
        with self.use_context() as vpool:
            self.formula, self.input_handles = get_formula(model, input_dim)
            self.input_ids = [vpool.id(h) for h in self.input_handles]
            for f in self.formula:
                f.clausify()
            self.output_ids = [vpool.id(f) for f in self.formula]
            # TODO/REMARK: formula represents output from second last layer
            # ie.: dimension is neuron_number, not class number
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fp_type = fp_type

    def check_model_with_data(self, model, data):
        with torch.no_grad(), self.use_context():
            model.train(False)

            for x, _ in data:
                x = x.to(self.fp_type).to(device)

                logit = model(x)
                p_logit = formula_as_pseudo_model(
                    formula=self.formula,
                    input_handles=self.input_handles,
                    output_dim=self.output_dim,
                )(x)
                assert logit.equal(p_logit)

    def check_model_with_truth_table(self, model):
        with torch.no_grad(), self.use_context():
            model.train(False)

            for x, _ in get_truth_table_loader(input_dim=self.input_dim):
                x = x.to(self.fp_type).to(device)

                logit = model(x)
                p_logit = formula_as_pseudo_model(
                    formula=self.formula,
                    input_handles=self.input_handles,
                    output_dim=self.output_dim,
                )(x)
                assert logit.equal(p_logit)

    def print(self, print_vpool=False):
        with self.use_context() as vpool:
            print("==== Formula ==== ")
            for f in self.formula:
                print(
                    f,
                    "...",
                    # f.simplified(), "...", f.clauses, "...", f.encoded, "...",
                    vpool.id(f),
                )

            print("==== Input Ids ==== ")
            print(self.input_ids)

            print("==== Output Ids ==== ")
            print(self.output_ids)

            if print_vpool:
                print("==== IDPool ====")
                for f, id in vpool.obj2id.items():
                    print(id, f)

    def calc_bound(self):
        return len(self.output_ids) // self.output_dim + 1

    def get_output_ids(self, class_id):
        step = len(self.output_ids) // self.output_dim
        start = (class_id - 1) * step
        return self.output_ids[start : start + step]

    def pairwise_comparisons(self, true_class, adj_class, inp=None):
        from pysat.card import CardEnc, EncType

        with self.use_context() as vpool:
            pos = self.get_output_ids(true_class)
            neg = [-a for a in self.get_output_ids(adj_class)]
            clauses = pos + neg
            print(clauses)
            # comp = CardEnc.atleast(
            #     lits=[5, 5, -12, -15], bound=3, encoding=EncType.totalizer, vpool=vpool
            # )
            comp = CardEnc.atleast(
                lits=clauses,
                bound=self.calc_bound(),
                encoding=EncType.totalizer,
                vpool=vpool,
            )
            clauses = comp.clauses
            print(comp.clauses)

            from pysat.solvers import Solver

            with Solver(bootstrap_with=clauses) as solver:
                solver.append_formula(comp)
                print(solver.solve(assumptions=inp))

    def try_it(self):
        from pysat.card import CardEnc, EncType

        self.pairwise_comparisons(1, 2, inp=[-1, -2, -3, -4, 5, -6, -7, -8])
        self.pairwise_comparisons(1, 3, inp=[-1, -2, -3, -4, 5, -6, -7, -8])

        # with self.use_context() as vpool:
        #     comp = CardEnc.atleast(
        #         lits=[5, 5, -2, -16], bound=3, encoding=EncType.totalizer, vpool=vpool
        #     )
        #     clauses = comp.clauses
        #     print(comp.clauses)

        #     from pysat.solvers import Solver

        #     with Solver(bootstrap_with=clauses) as solver:
        #         solver.append_formula(comp)
        #         print(solver.solve(assumptions=[-1, -2, -3, -4, 5, -6, -7, -8]))

    def check(self, model, data=None):
        if data != None:
            self.check_model_with_data(model, data)
        self.check_model_with_truth_table(model)
        # for h in self.input_handles:
        # print(id(h), id(Atom(h.name)))

    @contextmanager
    def use_context(self):
        hashable = id(self)
        prev = Formula._context
        try:
            Formula.set_context(hashable)
            yield Formula.export_vpool(active=True)
        finally:
            Formula.set_context(prev)

    def __del__(self):
        Formula.cleanup(id(self))
