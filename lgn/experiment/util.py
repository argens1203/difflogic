from pysat.card import EncType


def get_enc_type(enc_type):
    return {
        "pw": EncType.pairwise,
        "seqc": EncType.seqcounter,
        "cardn": EncType.cardnetwrk,
        "sortn": EncType.sortnetwrk,
        "tot": EncType.totalizer,
        "mtot": EncType.mtotalizer,
        "kmtot": EncType.kmtotalizer,
    }[enc_type]
