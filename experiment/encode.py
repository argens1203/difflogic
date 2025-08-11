from typing import Optional
from lgn.encoding import Encoder, SatEncoder, BddEncoder


class Encode:
    @staticmethod
    def get_encoding(model, enc_type, ctx, deduplication: Optional[str] = None):
        _Encoder = Encoder
        if deduplication == "sat":
            _Encoder = SatEncoder
        elif deduplication == "bdd":
            _Encoder = BddEncoder

        encoding = _Encoder().get_encoding(
            model,
            ctx.dataset,
            enc_type=enc_type,
        )

        if ctx.results is not None:
            ctx.results.store_encoding(encoding)
        if ctx.verbose:
            encoding.print()

        return encoding
