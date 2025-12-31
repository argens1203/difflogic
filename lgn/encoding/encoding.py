from typing import Any, Optional

from pysat.formula import Atom, Formula

from .pseudo_model import PseudoModel
from lgn.encoding.util import get_parts


class Encoding:
    def __init__(
        self,
        clauses: list[list[int]],
        eq_constraints: Any,
        input_ids: list[int],
        output_ids: list[Optional[int]],
        formula: list[Formula],
        special: dict[int, Atom],
        s_ctx: Any,
        e_ctx: Any,
    ) -> None:
        Dataset = e_ctx.get_dataset()
        self.parts = get_parts(Dataset, input_ids)

        self.clauses = clauses
        self.eq_constraints = eq_constraints

        self.Dataset = Dataset
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.formula = formula
        self.special: dict[int, Atom] = special

        self.s_ctx = s_ctx
        self.e_ctx = e_ctx

        self.input_dim = Dataset.get_input_dim()
        self.class_dim = Dataset.get_num_of_classes()

    def get_parts(self) -> list[tuple[int, int]]:
        return self.parts

    def get_cnf_clauses(self) -> list[list[int]]:
        return self.clauses

    def get_eq_constraints_clauses(self) -> list[list[int]]:
        return self.eq_constraints.clauses

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_dataset(self) -> Any:
        return self.Dataset

    def get_stats(self) -> dict[str, int]:
        return {
            "clauses_size": len(self.clauses),
            "eq_size": len(self.eq_constraints.clauses),
        }

    def get_output_ids(self, class_id: int) -> list[Optional[int]]:
        step = len(self.output_ids) // self.class_dim
        start = (class_id - 1) * step

        ret = self.output_ids[start : start + step]

        # Hacky way for SAT deduplication which produces no None in output_ids
        for idx in range(start, start + step):
            if idx in self.special:
                ret[idx - start] = None
        return ret

    def get_offset(self, class_id: int) -> int:
        step = len(self.output_ids) // self.class_dim
        start = (class_id - 1) * step
        return start

    def get_truth_value(self, idx: int) -> Optional[bool]:
        value = self.special.get(idx, None)
        if value is None:
            return None
        return value == Atom(True)

    def get_votes_per_cls(self) -> int:
        return len(self.output_ids) // self.class_dim

    def get_classes(self) -> list[int]:
        return list(range(1, self.class_dim + 1))

    def get_attribute_ranges(self) -> list[int]:
        return self.Dataset.get_attribute_ranges()

    def as_model(self) -> PseudoModel:
        model_args = {
            "class_dim": self.class_dim,
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
            "clauses": self.clauses + self.eq_constraints.clauses,
            "special": self.special,
        }
        return PseudoModel(**model_args)

    def print(self, print_vpool: bool = False) -> None:
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

            print("==== Truth Values ==== ")
            print("; ".join([f"{k}:{v}" for k, v in self.special.items()]))

    def get_vpool_size(self) -> int:
        with self.use_context() as vpool:
            return len(vpool.obj2id)

    def __del__(self) -> None:
        pass

    def use_context(self) -> Any:
        return self.s_ctx.use_vpool()

    def get_all_input_ids(self) -> list[int]:
        return self.input_ids

    def get_all_output_ids(self) -> list[Optional[int]]:
        return self.output_ids
