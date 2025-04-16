import logging
import torch
from typing import List
from contextlib import contextmanager

from pysat.formula import Formula, Atom, CNF, Or
from pysat.card import CardEnc, EncType

from difflogic import LogicLayer, GroupSum

from constant import device, Args

from lgn.dataset import AutoTransformer
from .bdd import BDDSolver

fp_type = torch.float32

logger = logging.getLogger(__name__)


def get_formula(model, input_dim, Dataset: AutoTransformer):
    # x = [Atom() for i in range(input_dim)]
    x = [Atom(i + 1) for i in range(input_dim)]
    inputs = x

    if Args["Deduplicate"]:
        print("Deduplicating...")
        solver = BDDSolver.from_inputs(inputs=x)
        solver.set_ohe(Dataset.get_attribute_ranges())
    else:
        print("Not deduplicating...")

    all = set()
    for i in x:
        all.add(i)

    for layer in model:
        assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
        if isinstance(layer, GroupSum):  # TODO: make get_formula for GroupSum
            continue
        x = layer.get_formula(x)
        if Args["Deduplicate"]:
            x = solver.deduplicate(x)
        for o in x:
            all.add(o)

    x[0] = Atom(False)
    return x, inputs


class Encoding:
    def __init__(
        self,
        model,
        input_dim,
        class_dim,
        Dataset: AutoTransformer,
        fp_type=fp_type,
        **kwargs,
    ):
        self.enc_type = kwargs.get("enc_type", EncType.totalizer)

        with self.use_context() as vpool:
            self.formula, self.input_handles = get_formula(model, input_dim, Dataset)
            self.input_ids = [vpool.id(h) for h in self.input_handles]
            self.cnf = CNF()
            self.output_ids = []
            self.special = dict()
            # adding the clauses to a global CNF
            for f in [
                Or(Atom(False), f.simplified()) for f in self.formula
            ]:  # TODO: Confirm this:
                f.clausify()
                self.cnf.extend(list(f)[:-1])
                logger.debug("Formula: %s", f)
                logger.debug("CNF Clauses: %s", f.clauses)
                logger.debug("Simplified: %s", f.simplified())
                logger.debug("CNF Clauses: %s", self.cnf.clauses)
                idx = 0
                if f.clauses[-1][1] is None:
                    self.special[idx] = f.simplified()
                self.output_ids.append(f.clauses[-1][1])
                idx += 1

                logger.debug("=== === === ===")
            logger.debug("CNF Clauses: %s", self.cnf.clauses)
            # REMARK: formula represents output from second last layer
            # ie.: dimension is neuron_number, not class number

        # NEW
        self.eq_constraints = CNF()
        self.parts = []
        with self.use_context() as vpool:
            start = 0
            logger.debug("full_input_ids: %s", self.input_ids)
            for step in Dataset.get_attribute_ranges():
                logger.debug("Step: %d", step)
                logger.debug("input_ids: %s", self.input_ids[start : start + step])
                part = self.input_ids[start : start + step]
                self.eq_constraints.extend(
                    CardEnc.equals(
                        lits=part,
                        vpool=vpool,
                        encoding=self.enc_type,
                    )
                )
                start += step
                self.parts.append(part)
        logger.debug("eq_constraints: %s", self.eq_constraints.clauses)
        # NEW

        self.input_dim = input_dim
        self.class_dim = class_dim
        self.fp_type = fp_type

        self.Dataset = Dataset

    def get_output_ids(self, class_id):
        step = len(self.output_ids) // self.class_dim
        start = (class_id - 1) * step
        return self.output_ids[start : start + step]

    def get_truth_value(self, idx):
        return self.special.get(idx, None)

    def get_votes_per_cls(self):
        return len(self.output_ids) // self.class_dim

    def get_classes(self):
        return list(range(1, self.class_dim + 1))

    def get_attribute_ranges(self):
        return self.Dataset.get_attribute_ranges()

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
                    f.simplified(),
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

    def get_enc_type(self):
        return self.enc_type

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
