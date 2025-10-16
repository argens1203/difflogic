from datetime import datetime
import csv
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
        self.enc_type_eq = get_enc_type(args.enc_type_eq)
        self.solver_type = args.solver_type
        self.fp_type = torch.float32

        self.cache_hit = {Cached_Key.SOLVER: 0}
        self.cache_miss = {Cached_Key.SOLVER: 0}
        self.deduplication = 0
        self.layer_seen = set()
        self.dedup_dict = dict()

        self.results.store_start_time()
        self.num_explanations = 0
        self.ohe_deduplication = []

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

    def inc_ohe_deduplication(self, ohe_from, ohe_to):
        self.ohe_deduplication.append((ohe_from, ohe_to))

    def store_clause(self, clauses: list[list[int]]):
        self.num_clauses = len(clauses)
        self.num_vars = max(abs(literal) for clause in clauses for literal in clause)

    def record_solving_stats(self, num_clauses, num_vars):
        self.solving_num_clauses = num_clauses
        self.solving_num_vars = num_vars

    def inc_num_explanations(self, num_explanations):
        self.num_explanations += num_explanations

    def output(self):
        if self.args.output == "display":
            self.display()
        elif self.args.output == "csv":
            self.to_csv()
        else:
            raise ValueError(f"Unknown output format: {self.args.output}")

    def get_headers(self):
        headers = [
            "ds",
            "input_dim",
            "# lay",
            "# neu",
            "acc",
            "enc",
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
            "strategy",
        ]
        return headers

    def get_data(self):
        number_of_gates = self.args.num_layers * self.args.num_neurons
        runtime = self.results.get_total_runtime()
        data = [
            [
                self.args.dataset,
                self.dataset.get_input_dim(),
                self.args.num_layers,
                self.args.num_neurons,
                self.results.test_acc,
                self.args.enc_type,
                self.args.enc_type_eq,
                self.args.solver_type,
                self.args.deduplicate,
                len(self.ohe_deduplication) if self.args.ohe_deduplication else "N/A",
                self.args.h_type,
                self.args.h_solver,
                number_of_gates,
                number_of_gates - self.deduplication,
                self.num_clauses,
                self.num_vars,
                "{:.2f}".format(self.solving_num_clauses),
                "{:.2f}".format(self.solving_num_vars),
                self.num_explanations,
                runtime,
                self.results.get_model_ready_time(),
                self.results.get_encoding_time(),
                self.results.get_explanation_time(),
                self.results.get_explanation_time() / self.num_explanations,
                humanfriendly.format_size(self.results.get_value("memory/encoding")),
                humanfriendly.format_size(self.results.get_value("memory/explanation")),
                self.args.strategy,
            ]
        ]
        return data

    def to_csv(self):
        data = self.get_data()
        headers = self.get_headers()
        with open("results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)

    def display(self):
        headers = self.get_headers()
        data = self.get_data()
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

    def get_enc_type_eq(self):
        return self.enc_type_eq

    def get_solver_type(self):
        return self.solver_type

    def get_fp_type(self):
        return self.fp_type

    def get_dataset(self):
        return self.dataset


class MultiContext:
    def __init__(self):
        self.data = []
        self.headers = None
        self.dedup_dict = []

    def add(self, ctx: Context):
        self.data.append(ctx.get_data()[0])
        self.dedup_dict.append(ctx.dedup_dict)
        self.headers = ctx.get_headers()

    @staticmethod
    def __unique_str_from_timestamp():
        return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]  # Up to milliseconds

    def to_csv(self, filename, with_timestamp=True):
        if with_timestamp:
            filename = f"{MultiContext.__unique_str_from_timestamp()}_{filename}"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.data)

    def display(self):
        headers = self.headers
        data = self.data
        print(tabulate(data, headers=headers, tablefmt="github"))

        # for dedup_dict in self.dedup_dict:
        #     for k in dedup_dict:
        #         print(f"Layer {k[0]} -> {k[1]}: {dedup_dict[k]}")
