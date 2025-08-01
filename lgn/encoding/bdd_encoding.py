from tqdm import tqdm
import logging

from pysat.formula import Formula, Atom

from difflogic import LogicLayer, GroupSum

from constant import Stats

from lgn.dataset import AutoTransformer
from lgn.deduplicator import BDDSolver, SolverWithDeduplication
from .encoding import Encoding

logger = logging.getLogger(__name__)


class BddEncoding(Encoding):
    def get_formula(
        self,
        model,
        input_dim,
        Dataset: AutoTransformer,
        deduplicator: SolverWithDeduplication,
        # TODO: second return is actually list[Atom] but cannot be defined as such
    ):
        with self.use_context():
            x: list[Formula] = [Atom(i + 1) for i in range(input_dim)]
            inputs = x

            logger.debug("Deduplicating...")
            solver = BDDSolver.from_inputs(inputs=x)
            solver.set_ohe(Dataset.get_attribute_ranges())

            all = set()
            for i in x:
                all.add(i)
            Stats["deduplication"] = 0

            for i, layer in enumerate(model):
                logger.debug("Layer %d: %s", i, layer)
                assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
                if isinstance(layer, GroupSum):  # TODO: make get_formula for GroupSum
                    continue
                x = layer.get_formula(x)
                for idx in tqdm(range(len(x))):
                    x[idx] = solver.deduplicate(x[idx], all)
                    all.add(x[idx])

        return x, inputs
