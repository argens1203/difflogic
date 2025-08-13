import logging
import tracemalloc
from typing import Callable

from .logging import setup_logger
from .util import seed_all, get_results

from lgn.dataset import load_dataset


class Cached_Key:
    SOLVER = "solver"


class Context:
    def __init__(self, args):
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

    def debug(self, l: Callable):
        if self.verbose == "debug":
            l()

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

    def __del__(self):
        self.logger.debug("Cache Hit: %s", str(self.cache_hit))
        self.logger.debug("Cache Miss: %s", str(self.cache_miss))
        self.logger.debug("Deduplication: %s", str(self.deduplication))
