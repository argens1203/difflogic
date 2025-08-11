from attr import dataclass
from typing import Union
from experiment.util.args.encoding_args import EncodingArgs
from experiment.util.args.pysat_args import PySatArgs
from lgn.encoding import Encoder, SatEncoder, BddEncoder
from .context import Context
from .util import get_enc_type


@dataclass
class Args(EncodingArgs, PySatArgs):
    pass


class Encode:
    @staticmethod
    def get_encoding(model, args: Args, ctx: Context):
        _Encoder = Encoder
        if args.deduplicate == "sat":
            _Encoder = SatEncoder
        elif args.deduplicate == "bdd":
            _Encoder = BddEncoder

        encoding = _Encoder().get_encoding(
            model,
            ctx.dataset,
            enc_type=get_enc_type(args.enc_type),
        )

        if ctx.results is not None:
            ctx.results.store_encoding(encoding)
        if ctx.verbose:
            encoding.print()

        return encoding
