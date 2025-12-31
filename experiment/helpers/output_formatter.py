"""Output formatting for experiment results."""

import csv
from typing import Any, TYPE_CHECKING

from tabulate import tabulate
import humanfriendly

if TYPE_CHECKING:
    from .context import Context


HEADERS = [
    "ds",
    "input_dim",
    "# lay",
    "# neu",
    "acc",
    "enc_at_l",
    "enc_eq",
    "solver",
    "ddup",
    "ohe-ddup",
    "h_type",
    "h_solver",
    "# gates",
    "# gates_f",
    "# cl",
    "# var",
    "# avg_cl_s",
    "# avg_var_s",
    "# expl",
    "run_t",
    "t_Model",
    "t_Encoding",
    "t_Explain",
    "t/Exp",
    "m_enc",
    "m_expl",
    "dedup_strat",
    "exp_strat",
    "proc_rounds",
]


class OutputFormatter:
    """Formats experiment results for display or CSV export."""

    def __init__(self, ctx: "Context") -> None:
        self._ctx = ctx

    def get_headers(self) -> list[str]:
        """Get column headers for output."""
        return HEADERS

    def get_data(self) -> list[list[Any]]:
        """Get formatted data row for output."""
        ctx = self._ctx
        args = ctx.args
        results = ctx.results
        dataset = ctx.dataset
        dedup_tracker = ctx.dedup_tracker
        solving_stats = ctx.solving_stats

        number_of_gates = args.num_layers * args.num_neurons
        runtime = results.get_total_runtime()

        ohe_count = (
            len(dedup_tracker.ohe_deduplication)
            if args.ohe_deduplication
            else "N/A"
        )

        data = [
            [
                args.dataset,
                dataset.get_input_dim(),
                args.num_layers,
                args.num_neurons,
                results.test_acc,
                args.enc_type_at_least,
                args.enc_type_eq,
                args.solver_type,
                args.deduplicate,
                ohe_count,
                args.h_type,
                args.h_solver,
                number_of_gates,
                number_of_gates - dedup_tracker.count,
                solving_stats.num_clauses,
                solving_stats.num_vars,
                "{:.2f}".format(solving_stats.get_avg_clauses()),
                "{:.2f}".format(solving_stats.get_avg_vars()),
                ctx.num_explanations,
                runtime,
                results.get_model_ready_time(),
                results.get_encoding_time(),
                results.get_explanation_time(),
                results.get_explanation_time() / ctx.num_explanations,
                humanfriendly.format_size(results.get_value("memory/encoding")),
                humanfriendly.format_size(results.get_value("memory/explanation")),
                args.strategy,
                args.explain_algorithm,
                ctx.get_process_rounds(),
            ]
        ]
        return data

    def to_csv(self, filename: str = "results.csv") -> None:
        """Write results to CSV file."""
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.get_headers())
            writer.writerows(self.get_data())

    def display(self) -> None:
        """Display results as formatted table."""
        print(tabulate(self.get_data(), headers=self.get_headers(), tablefmt="github"))
        self._ctx.dedup_tracker.print_summary()
