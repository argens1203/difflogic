from tqdm import tqdm
import logging

from pysat.formula import Atom

from difflogic import LogicLayer, GroupSum

from constant import Stats

from lgn.dataset import AutoTransformer
from ..deduplicator import SolverWithDeduplication
from .encoder import Encoder

logger = logging.getLogger(__name__)


class SatEncoder(Encoder):
    def get_formula(
        self,
        model,
        input_dim,
        Dataset: AutoTransformer,
        deduplicator: SolverWithDeduplication,
    ):
        with self.use_context():
            x = [Atom(i + 1) for i in range(input_dim)]
            inputs = x

            logger.debug("Deduplicating with SAT solver ...")
            all = set()
            for i in x:
                all.add(i)
            Stats["deduplication"] = 0

            for layer in model:
                assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
                if isinstance(layer, GroupSum):
                    continue
                x = layer.get_formula(x)
                assert x is not None, "Layer returned None"
                for idx in tqdm(range(len(x))):
                    x[idx] = deduplicator.deduplicate(x[idx], all)
                    all.add(x[idx])
                    assert x[idx] is not None, "Deduplicator returned None"

        return x, inputs
