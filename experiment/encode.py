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
        _Encoder = Encoder
        if args.deduplicate == "sat":
            logger.info("Using SAT Encoder")
            _Encoder = SatEncoder
        elif args.deduplicate == "bdd":
            logger.info("Using BDD Encoder")
            _Encoder = BddEncoder

        encoding = _Encoder().get_encoding(
            model,
            ctx.dataset,
            enc_type=get_enc_type(args.enc_type),
        )

        if ctx.results is not None:
            ctx.results.store_encoding(encoding)

        ctx.debug(encoding.print)

        return encoding
