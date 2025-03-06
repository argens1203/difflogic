import torch
from constant import device
from .util import formula_as_pseudo_model, get_truth_table_loader
from pysat.formula import Formula, Atom
from difflogic import LogicLayer, GroupSum

fp_type = torch.float32


def get_formula(model, input_dim):
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
    def from_model(model, input_dim, output_dim, fp_type=fp_type):
        formula, input_handles = get_formula(model, input_dim)
        return PseudoModel(
            formula=formula,
            input_handles=input_handles,
            input_dim=input_dim,
            output_dim=output_dim,
            fp_type=fp_type,
        )

    def __init__(self, formula, input_handles, input_dim, output_dim, fp_type=fp_type):
        self.formula = formula
        self.input_handles = input_handles
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fp_type = fp_type
        # print(id(self))
        # input()

    def check_model_with_data(self, model, data):
        with torch.no_grad():
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
        with torch.no_grad():
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
        # Formula.cleanup()
        print("formula: ", self.formula)
        for f in self.formula:
            f.clausify()
        self.clausified = [f.clauses for f in self.formula]
        print("clausified: ", self.clausified)
        print("input_handles:", self.input_handles)

    def check(self, model, data=None):
        if data != None:
            self.check_model_with_data(model, data)
        self.check_model_with_truth_table(model)
