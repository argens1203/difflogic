import logging
import torch
from contextlib import contextmanager

from pysat.formula import Formula, Atom, CNF
from difflogic import LogicLayer, GroupSum

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
            cnf = CNF()
            for f in self.formula:
                f.clausify()
                cnf.extend(f.simplified())
            self.cnf = cnf
            self.output_ids = [vpool.id(f) for f in self.formula]
            # TODO/REMARK: formula represents output from second last layer
            # ie.: dimension is neuron_number, not class number
        self.input_dim = input_dim
        self.class_dim = class_dim
        self.fp_type = fp_type

    def get_output_ids(self, class_id):
        step = len(self.output_ids) // self.class_dim
        start = (class_id - 1) * step
        return self.output_ids[start : start + step]

    def get_votes_per_cls(self):
        return len(self.output_ids) // self.class_dim

    def get_classes(self):
        return list(range(1, self.class_dim + 1))

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
        """
        This methods returns the class label given the input. Or returns the votes for each class if logit is set to True.

        param: x: input
        type: torch.Tensor (batch_size, input_dim)

        param: logit: if True, returns votes for each class
        type: bool

        returns: class label or votes
        rtype: torch.Tensor (batch_size) or torch.Tensor (batch_size, no_of_classes)
        """
        bool_outputs = (
            torch.tensor(
                [
                    [
                        (
                            0
                            if f.simplified(
                                assumptions=[
                                    ~inp if feat == 0 else inp
                                    for feat, inp in zip(features, self.input_handles)
                                ]
                            )
                            # TODO: better way of checking?
                            == Atom(False)
                            else 1
                        )
                        for f in self.formula
                    ]
                    for features in x
                ]
            )
            .to(device)
            .int()
        )
        votes = PseudoModel._count_votes(bool_outputs, self.class_dim)

        if logit:
            return votes

        cls_label = votes.argmax().int()
        logger.debug("Class Label: %d", cls_label)
        return cls_label

    def _count_votes(logits: torch.Tensor, no_of_classes: int):
        """
        This methods takes a binary input and returns the number of 1s in each class.

        param: logits: binary input
        type: torch.Tensor (batch_size, no_of_classes * votes_per_class)

        returns: votes
        rtype: torch.Tensor (batch_size, no_of_classes)
        """
        assert logits.shape[1] % no_of_classes == 0

        votes_per_class = logits.shape[1] // no_of_classes
        reshaped = logits.view(len(logits), -1, votes_per_class)

        return reshaped.sum(dim=2)
