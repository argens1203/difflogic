import logging
import tracemalloc
from experiment.util import setup_logger, seed_all
from lgn.dataset import load_dataset
from .util import get_results

logger = logging.getLogger(__name__)


class Cached_Key:
    SOLVER = "solver"


class Context:
    def __init__(self, args):
        setup_logger(args)
        seed_all(args.seed)

        self.logger = logging.getLogger()

        eid = args.experiment_id if args.experiment_id is not None else 0
        self.results = get_results(eid, args)
        self.train_loader, self.test_loader, self.get_raw, self.dataset = load_dataset(
            args
        )
        self.verbose = args.verbose

        self.cache_hit = {Cached_Key.SOLVER: 0}
        self.cache_miss = {Cached_Key.SOLVER: 0}
        self.deduplication = 0

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
        logger.debug("Cache Hit: %s", str(self.cache_hit))
        logger.debug("Cache Miss: %s", str(self.cache_miss))
        logger.debug("Deduplication: %s", str(self.deduplication))
