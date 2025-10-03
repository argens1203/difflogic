from .pseudo_model import PseudoModel
from lgn.encoding.util import get_parts


class Encoding:
    def __init__(
        self,
        clauses,
        eq_constraints,
        input_ids,
        output_ids,
        formula,
        special,
        s_ctx,
        e_ctx,
    ):
        Dataset = e_ctx.get_dataset()
        self.parts = get_parts(Dataset, input_ids)

        self.clauses = clauses
        self.eq_constraints = eq_constraints

        self.Dataset = Dataset
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.formula = formula
        self.special = special

        self.s_ctx = s_ctx
        self.e_ctx = e_ctx

        self.input_dim = Dataset.get_input_dim()
        self.class_dim = Dataset.get_num_of_classes()

    def get_parts(self):
        return self.parts

    def get_cnf_clauses(self):
        return self.clauses

    def get_eq_constraints_clauses(self):
        return self.eq_constraints.clauses

    def get_input_dim(self):
        return self.input_dim

    def get_dataset(self):
        return self.Dataset

    def get_stats(self):
        return {
            "clauses_size": len(self.clauses),
            "eq_size": len(self.eq_constraints.clauses),
        }

    def get_output_ids(self, class_id):
        step = len(self.output_ids) // self.class_dim
        start = (class_id - 1) * step

        ret = self.output_ids[start : start + step]

        # Hacky way for SAT deduplication which produces no None in output_ids
        for idx in range(start, start + step):
            if idx in self.special:
                ret[idx - start] = None
        return ret

    def get_offset(self, class_id):
        step = len(self.output_ids) // self.class_dim
        start = (class_id - 1) * step
        return start

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
            "class_dim": self.class_dim,
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
            "clauses": self.clauses + self.eq_constraints.clauses,
            "special": self.special,
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

    def get_vpool_size(self):
        with self.use_context() as vpool:
            return len(vpool.obj2id)

    def __del__(self):
        pass

    def use_context(self):
        return self.s_ctx.use_vpool()
