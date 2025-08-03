from contextlib import contextmanager

from pysat.formula import Formula
from .pseudo_model import PseudoModel


class Encoding:
    def __init__(
        self,
        parts,
        cnf,
        eq_constraints,
        input_dim,
        fp_type,
        Dataset,
        class_dim,
        input_ids,
        output_ids,
        formula,
        input_handles,
        special,
        enc_type,
        vpool_context,
    ):
        self.parts = parts
        self.cnf = cnf
        self.eq_constraints = eq_constraints
        self.input_dim = input_dim
        self.fp_type = fp_type
        self.Dataset = Dataset
        self.class_dim = class_dim
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.formula = formula
        self.input_handles = input_handles
        self.special = special
        self.enc_type = enc_type
        self.vpool_context = vpool_context

    def get_parts(self):
        return self.parts

    def get_cnf_clauses(self):
        return self.cnf.clauses

    def get_eq_constraints_clauses(self):
        return self.eq_constraints.clauses

    def get_input_dim(self):
        return self.input_dim

    def get_fp_type(self):
        return self.fp_type

    def get_dataset(self):
        return self.Dataset

    def get_stats(self):
        return {
            "cnf_size": len(self.cnf.clauses),
            "eq_size": len(self.eq_constraints.clauses),
        }

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
        model_args = {
            "input_handles": self.input_handles,
            "formula": self.formula,
            "class_dim": self.class_dim,
        }
        return PseudoModel(**model_args)

    def print(self, print_vpool=False):
        with self.use_context() as vpool:
            print("self_id", id(self))
            print("vpool_id", id(vpool))
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
                for f, e in vpool.obj2id.items():
                    print(e, f)

    def get_enc_type(self):
        return self.enc_type

    @contextmanager
    def use_context(self):
        prev = Formula._context
        try:
            Formula.set_context(self.vpool_context)
            yield Formula.export_vpool(active=True)
        finally:
            Formula.set_context(prev)

    def __del__(self):
        pass
        # Formula.cleanup(self.vpool_context)
