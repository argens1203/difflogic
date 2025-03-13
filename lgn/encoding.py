import logging
import torch
from contextlib import contextmanager

from pysat.formula import Formula, Atom, IDPool
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver
from difflogic import LogicLayer, GroupSum

from constant import device
from .util import formula_as_pseudo_model, get_truth_table_loader, feat_to_input

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

    # Returns the votes, or logits, before argmax of prediction
    def predict_votes(self, x):
        with self.use_context():
            return formula_as_pseudo_model(
                formula=self.formula,
                input_handles=self.input_handles,
                class_dim=self.class_dim,
            )(x)

    def check_model_with_data(self, model, data):
        with torch.no_grad(), self.use_context():
            model.train(False)
            logger.debug("Checking model with data")
            for x, _ in data:
                x = x.to(self.fp_type).to(device)

                logit = model(x)
                p_logit = self.predict_votes(x)
                assert logit.equal(p_logit)

    def check_model_with_truth_table(self, model):
        with torch.no_grad(), self.use_context():
            model.train(False)

            for x, _ in get_truth_table_loader(input_dim=self.input_dim):
                x = x.to(self.fp_type).to(device)

                logit = model(x)
                p_logit = self.predict_votes(x)
                assert logit.equal(p_logit)

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

    def calc_bound(self, true_class, adj_class):
        # TODO: CHECK
        # If true_class < adj_class, then adj class need one more vote to be selected
        # Else the number of vote needed is output neuron / number of classes
        if true_class < adj_class:
            return len(self.output_ids) // self.class_dim + 1
        return len(self.output_ids) // self.class_dim

    def get_output_ids(self, class_id):
        step = len(self.output_ids) // self.class_dim
        start = (class_id - 1) * step
        return self.output_ids[start : start + step]

    def pairwise_comparisons(self, true_class, adj_class, inp=None):
        logger.debug(
            "Pairwise Comparisons (%d > %d)",
            true_class,
            adj_class,
        )

        with self.use_context() as vpool:
            pos = self.get_output_ids(adj_class)
            neg = [-a for a in self.get_output_ids(true_class)]
            clauses = pos + neg  # Sum of X_i - Sum of X_pi_i > bounding number
            logger.debug("Lit: %s", str(clauses))

            bound = self.calc_bound(true_class, adj_class)
            logger.debug("Bound: %d", bound)
            comp = CardEnc.atleast(
                lits=clauses,
                bound=bound,
                encoding=EncType.totalizer,
                vpool=vpool,
            )

            clauses = comp.clauses
            logger.debug("Card Encoding Clauses: %s", str(comp.clauses))

            # Enumerate all clauses
            with Solver(bootstrap_with=clauses) as solver:
                # TODO: CHECK
                # Check if it is satisfiable under cardinatlity constraint
                solver.append_formula(comp)
                result = solver.solve(assumptions=inp)
                logger.debug("Satisfiable: %s", result)
                return result

        assert False, "Pairwise comparison error"

    def check(self, model, data=None):
        if data != None:
            self.check_model_with_data(model, data)
        self.check_model_with_truth_table(model)

    def explain(self, feat):
        logger.debug("feat: %s", feat)
        inp = feat_to_input(feat)
        logger.debug("inp: %s", inp)
        logger.info("Explaining: %s", inp)
        votes = self.predict_votes(torch.tensor([inp]))
        logger.debug("Votes: %s", votes)
        true_class = votes.argmax().int() + 1
        logger.info("Predicted Class - %s", true_class)

        assert self.uniquely_satisfied_by(inp, true_class)

        reduced = self.reduce(inp, true_class)
        logger.info("Final reduced: %s", reduced)

    def reduce(self, inp, true_class):
        temp = inp.copy()
        for feature in inp:
            logger.info("Testing removal of %d", feature)
            temp.remove(feature)
            if self.uniquely_satisfied_by(inp=temp, true_class=true_class):
                continue
            else:
                temp.append(feature)
        return temp

    def uniquely_satisfied_by(
        self,
        inp,
        true_class,  # true_class as in not true class of data, but that predicted by model
    ):  # Return true means that only the true_class can satisfy all contraints given the input
        for cls in range(1, self.class_dim + 1):  # TODO: use more consistent ways
            if cls == true_class:
                continue
            if (
                self.pairwise_comparisons(true_class=true_class, adj_class=cls, inp=inp)
                is True
            ):
                logger.debug("Satisfied by %d", cls)
                return False
        return True

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
