import torch
from constant import *
from .util import formula_as_pseudo_model, get_truth_table_loader

fp_type = torch.float32


class PseudoModel:
    def __init__(self, formula, input_handles, input_dim, output_dim, fp_type=fp_type):
        self.formula = formula
        self.input_handles = input_handles
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fp_type = fp_type

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
        print(self.formula)
        print(self.input_handles)

    def check(self, model, data=None):
        if data != None:
            self.check_model_with_data(model, data)
        self.check_model_with_truth_table(model)
