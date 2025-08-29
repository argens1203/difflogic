from tqdm import tqdm
import logging

from difflogic import LogicLayer, GroupSum
from pysat.formula import Formula, Atom

from experiment.helpers.ordered_set import OrderedSet

from .encoder import Encoder
from .bdd_deduplicator import BDDSolver

from lgn.dataset import AutoTransformer

logger = logging.getLogger(__name__)


class BddEncoder(Encoder):
    def get_formula(
        self,
        model,
        input_dim,
        Dataset: AutoTransformer,
        # TODO: second return is actually list[Atom] but cannot be defined as such
    ):
        with self.use_context():
            x: list[Formula] = [Atom(i + 1) for i in range(input_dim)]
            inputs = x

            logger.debug("Deduplicating...")
            solver = BDDSolver.from_inputs(inputs=x, e_ctx=self.e_ctx)
            solver.set_ohe(Dataset.get_attribute_ranges())

            all = OrderedSet()
            #  TODO: uncomment this?
            for i in x:
                all.add((0, i))

            for i, layer in enumerate(model):
                logger.debug("Layer %d: %s", i, layer)
                assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
                if isinstance(layer, GroupSum):  # TODO: make get_formula for GroupSum
                    continue
                x = layer.get_formula(x)

                self.e_ctx.debug(lambda: print("Before deduplication:"))
                for idx, f in enumerate(x):
                    self.e_ctx.debug(lambda: print(idx, ":", f))

                for idx in tqdm(range(len(x))):
                    x[idx] = solver.deduplicate(x[idx], all, i + 1)
                    all.add((i + 1, x[idx]))

                self.e_ctx.debug(lambda: print("After deduplication:"))
                for idx, f in enumerate(x):
                    self.e_ctx.debug(lambda: print(idx, ":", f))

        del solver
        return x, inputs
