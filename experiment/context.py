import logging
from lgn.dataset import (
    new_load_dataset as load_dataset,
)
from .util import get_results


class Context:
    def __init__(self, args):
        self.logger = logging.getLogger()

        eid = args.experiment_id if args.experiment_id is not None else 0
        self.results = get_results(eid, args)
        self.train_loader, self.test_loader, self.get_raw, self.dataset = load_dataset(
            args
        )
        self.verbose = args.verbose
