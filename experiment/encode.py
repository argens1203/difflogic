import logging

from attr import dataclass
from experiment.args import EncodingArgs, PySatArgs
from lgn.encoding import Encoder, SatEncoder, BddEncoder
from .helpers import Context, get_enc_type

logger = logging.getLogger(__name__)


@dataclass
class Args(EncodingArgs, PySatArgs):
    pass


class Encode:
    @staticmethod
    def get_encoding(model, args: Args, ctx: Context):
        _Encoder = Encoder(e_ctx=ctx)  # Default Encoder (no deduplication)
        if args.deduplicate == "sat":
            logger.info("Using SAT Encoder...")
            _Encoder = SatEncoder(e_ctx=ctx)
            _Encoder.strategy = args.strategy
            _Encoder.deduplicate_ohe = args.ohe_deduplication
            assert args.strategy in ["full", "b_full", "parent"]

        elif args.deduplicate == "bdd":
            logger.info("Using BDD Encoder...")
            _Encoder = BddEncoder(e_ctx=ctx)

        encoding = _Encoder.get_encoding(
            model,
            ctx.dataset,
        )

        # input("Press Enter to continue...")

        if ctx.results is not None:
            ctx.results.store_encoding(encoding)

        ctx.debug(encoding.print)
        ctx.results.store_encoding_ready_time()
        return encoding
