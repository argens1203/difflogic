from tqdm import tqdm
import logging
import torch

from pysat.formula import Atom

from difflogic import LogicLayer, GroupSum

from constant import Stats

from lgn.dataset import AutoTransformer
from ..deduplicator import SatDeduplicator
from .encoder import Encoder
from ..context import Context
from ..encoding import Encoding

from pysat.card import EncType

logger = logging.getLogger(__name__)

fp_type = torch.float32


class SatEncoder(Encoder):
    def get_static(self, model, Dataset: AutoTransformer, fp_type=fp_type, **kwargs):
        first_encoding = Encoder().get_static(model, Dataset, fp_type=fp_type, **kwargs)
        deduplicator = SatDeduplicator(first_encoding)

        enc_type = kwargs.get("enc_type", EncType.totalizer)
        self.context = Context()

        input_dim = Dataset.get_input_dim()
        class_dim = Dataset.get_num_of_classes()

        formula, input_handles = self.get_formula(
            model, input_dim, Dataset, deduplicator=deduplicator
        )
        input_ids, cnf, output_ids, special = self.populate_clauses(
            input_handles=input_handles, formula=formula
        )

        eq_constraints, parts = self.initialize_ohe(Dataset, input_ids, enc_type)
        deduplicator.delete()

        return Encoding(
            parts=parts,
            cnf=cnf,
            eq_constraints=eq_constraints,
            input_dim=input_dim,
            fp_type=fp_type,
            Dataset=Dataset,
            class_dim=class_dim,
            input_ids=input_ids,
            output_ids=output_ids,
            formula=formula,
            input_handles=input_handles,
            special=special,
            enc_type=enc_type,
            context=self.context,
        )

    def get_formula(
        self,
        model,
        input_dim,
        Dataset: AutoTransformer,
        deduplicator: SatDeduplicator,
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
