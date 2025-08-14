import logging
import tracemalloc
from contextlib import contextmanager
from typing import Callable
from tabulate import tabulate
import humanfriendly
from .logging import setup_logger
from .util import seed_all, get_results

from lgn.dataset import load_dataset


class Cached_Key:
    SOLVER = "solver"


class Context:
    def __init__(self, args):
        self.args = args
        setup_logger(args)
        seed_all(args.seed)

        self.logger = logging.getLogger(__name__)

        eid = args.experiment_id if args.experiment_id is not None else 0
        self.results = get_results(eid, args)
        self.train_loader, self.test_loader, self.get_raw, self.dataset = load_dataset(
            args
        )
        self.verbose = args.verbose

        self.cache_hit = {Cached_Key.SOLVER: 0}
        self.cache_miss = {Cached_Key.SOLVER: 0}
        self.deduplication = 0

        self.results.store_start_time()
        self.num_explanations = 0

    def debug(self, l: Callable):
        if self.verbose == "debug":
            l()

    @contextmanager
    def use_memory_profile(self):
        try:
            tracemalloc.start()
            yield lambda label: self.store_memory_peak(label)
        finally:
            tracemalloc.stop()

    def store_memory_peak(self, label=None):
        current, peak = tracemalloc.get_traced_memory()  # in Bytes
        # print("current", current)
        # print("peak", peak)
        self.results.store_custom(f"memory/{label}", peak)
        tracemalloc.reset_peak()

    def start_memory_usage(self):
        tracemalloc.start()

    def end_memory_usage(self):
        tracemalloc.stop()

    def get_memory_usage(self, label=None):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    def inc_cache_hit(self, flag: str):
        self.cache_hit[flag] += 1

    def inc_cache_miss(self, flag: str):
        self.cache_miss[flag] += 1

    def reset_deduplication(self):
        self.deduplication = 0

    def inc_deduplication(self):
        self.deduplication += 1

    def store_num_clauses(self, num_clauses):
        self.num_clauses = num_clauses

    def inc_num_explanations(self, num_explanations):
        self.num_explanations += num_explanations

    def display(self):
        number_of_gates = self.args.num_layers * self.args.num_neurons
        runtime = self.results.get_total_runtime()

        headers = [
            "ds",
            "input_dim",
            "# lay",
            "# neu",
            "acc",
            "enc",
            "solver",
            "ddup",
            "# gates",
            "# gates_f",
            "# cl",
            "# expl",
            "run_t",
            "t_Model",
            "t_Encoding",
            "t_Explain",
            "t/Exp",
            "m_enc",
            "m_expl",
        ]
        data = [
            [
                self.args.dataset,
                self.dataset.get_input_dim(),
                self.args.num_layers,
                self.args.num_neurons,
                self.results.test_acc,
                self.args.enc_type,
                self.args.solver_type,
                self.args.deduplicate,
                number_of_gates,
                number_of_gates - self.deduplication,
                self.num_clauses,
                self.num_explanations,
                runtime,
                self.results.get_model_ready_time(),
                self.results.get_encoding_time(),
                self.results.get_explanation_time(),
                self.results.get_explanation_time() / self.num_explanations,
                humanfriendly.format_size(self.results.get_value("memory/encoding")),
                humanfriendly.format_size(self.results.get_value("memory/explanation")),
            ]
        ]
        print(tabulate(data, headers=headers, tablefmt="github"))

    def __del__(self):
        self.logger.debug("Cache Hit: %s", str(self.cache_hit))
        self.logger.debug("Cache Miss: %s", str(self.cache_miss))
        self.logger.debug("Deduplication: %s", str(self.deduplication))
