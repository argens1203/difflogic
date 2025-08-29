import logging
import tracemalloc
from contextlib import contextmanager
from typing import Callable
from tabulate import tabulate
import humanfriendly
import torch
from .logging import setup_logger
from .util import seed_all, get_results, get_enc_type
from lgn.dataset import load_dataset
from collections import OrderedDict


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
        self.enc_type = get_enc_type(args.enc_type)
        self.solver_type = args.solver_type
        self.fp_type = torch.float32

        self.cache_hit = {Cached_Key.SOLVER: 0}
        self.cache_miss = {Cached_Key.SOLVER: 0}
        self.deduplication = 0
        self.layer_seen = set()
        self.dedup_dict = dict()

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
        self.dedup_dict = OrderedDict()

    def inc_deduplication(self, curr_layer, target_layer):
        CONSTANT = "Constant"
        INPUT = "Input"

        if curr_layer not in self.layer_seen:
            self.layer_seen.add(curr_layer)
            self.dedup_dict[(curr_layer, CONSTANT)] = 0
            self.dedup_dict[(curr_layer, INPUT)] = 0
            for k in range(1, curr_layer + 1):
                self.dedup_dict[(curr_layer, k)] = 0
        # curr_layer = 1-based
        # target_layer = 1-based, 0 = input, -1 = constants

        if target_layer == -1:
            target_layer = CONSTANT
        if target_layer == 0:
            target_layer = INPUT
        self.deduplication += 1
        self.dedup_dict[(curr_layer, target_layer)] = (
            self.dedup_dict.get((curr_layer, target_layer), 0) + 1
        )

    def store_clause(self, clause: list[list[int]]):
        self.num_clauses = len(clause)
        self.num_vars = max(abs(literal) for clause in clause for literal in clause)

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
            "# var",
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
                self.num_vars,
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
        self.print_dedup_dict()

    def print_dedup_dict(self):
        for k in self.dedup_dict:
            print(f"Layer {k[0]} -> {k[1]}: {self.dedup_dict[k]}")

    def __del__(self):
        self.logger.debug("Cache Hit: %s", str(self.cache_hit))
        self.logger.debug("Cache Miss: %s", str(self.cache_miss))
        self.logger.debug("Deduplication: %s", str(self.deduplication))

    def get_enc_type(self):
        return self.enc_type

    def get_solver_type(self):
        return self.solver_type

    def get_fp_type(self):
        return self.fp_type

    def get_dataset(self):
        return self.dataset
