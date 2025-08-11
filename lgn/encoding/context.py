import logging
import torch

from contextlib import contextmanager

from pysat.formula import Formula

fp_type = torch.float32

logger = logging.getLogger(__name__)


class Context:
    def __init__(self):
        self.vpool_context = id(self)

    def get_vpool_context(self):
        return self.vpool_context

    @contextmanager
    def use_vpool(self):
        prev = Formula._context
        try:
            Formula.set_context(self.vpool_context)
            yield Formula.export_vpool(active=True)
        finally:
            Formula.set_context(prev)

    def __del__(self):
        self.delete()

    def delete(self):
        Formula.cleanup(self.vpool_context)
