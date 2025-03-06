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
        with self.use_context():
            self.formula, self.input_handles = get_formula(model, input_dim)
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

    def print(self):
        with self.use_context() as vpool:
            # Formula.cleanup()

            print("formula: ")
            for f in self.formula:
                f.clausify()
                print(f.simplified(), "...", f.clauses, "...", vpool.id(f))
                # print()
            # self.clausified = [f.clauses for f in self.formula]
            # print("clausified: ", self.clausified)
            print("input_handles:", self.input_handles)

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
