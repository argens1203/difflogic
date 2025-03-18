import logging
import torch
from contextlib import contextmanager

from pysat.formula import Formula, Atom
from difflogic import LogicLayer, GroupSum

from lgn.util import summed_on
from constant import device

fp_type = torch.float32

logger = logging.getLogger(__name__)


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


class Encoding:
    def __init__(self, model, input_dim, class_dim, fp_type=fp_type):
        with self.use_context() as vpool:
            self.formula, self.input_handles = get_formula(model, input_dim)
            self.input_ids = [vpool.id(h) for h in self.input_handles]
            for f in self.formula:
                f.clausify()
            self.output_ids = [vpool.id(f) for f in self.formula]
            # TODO/REMARK: formula represents output from second last layer
            # ie.: dimension is neuron_number, not class number
        self.input_dim = input_dim
        self.class_dim = class_dim
        self.fp_type = fp_type

    def print(self, print_vpool=False):
        with self.use_context() as vpool:
            print("==== Formula ==== ")
            for f in self.formula:
                print(
                    (str(vpool.id(f)) + ")").ljust(4),
                    f,
                    # f.simplified(), "...", f.clauses, "...", f.encoded, "...",
                )

            print("==== Input Ids ==== ")
            print(self.input_ids)

            print("==== Output Ids ==== ")
            print(self.output_ids)

            if print_vpool:
                print("==== IDPool ====")
                for f, id in vpool.obj2id.items():
                    print(id, f)

    def get_output_ids(self, class_id):
        step = len(self.output_ids) // self.class_dim
        start = (class_id - 1) * step
        return self.output_ids[start : start + step]

    def as_model(self):
        """
        The method returns a callable model which predicts the class labels given instances.

        :return: labels
        :rtype: torch.Tensor (batch_size)

        Example:

        .. code-block:: python
            >>> x = Torch.tensor([[1, 0, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0, 0, 1]])
            >>> encoding.as_model()(x)
            'Torch.tensor([0, 1])'
        """
        model_args = {
            "input_handles": self.input_handles,
            "formula": self.formula,
            "class_dim": self.class_dim,
        }
        return PseudoModel(**model_args)

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


class PseudoModel:
    def __init__(self, input_handles, formula, class_dim):
        self.input_handles = input_handles
        self.formula = formula
        self.class_dim = class_dim

    def __call__(self, x: torch.Tensor, logit=False):
        simplified = [
            [
                (
                    0
                    if f.simplified(
                        assumptions=[
                            ~inp if feat == 0 else inp
                            for feat, inp in zip(xs, self.input_handles)
                        ]
                    )
                    # TODO: better way of checking?
                    == Atom(False)
                    else 1
                )
                for f in self.formula
            ]
            for xs in x
        ]
        summed = (
            torch.tensor([summed_on(inter, self.class_dim) for inter in simplified])
            .to(device)
            .int()
        )
        if logit:
            return summed

        cls_label = summed.argmax().int()
        logger.debug("Class Label: %d", cls_label)
        return cls_label
