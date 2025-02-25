def model_print(model):
    for x in model:
        print(x)


from difflogic import LogicLayer, GroupSum
from pysat.formula import *


def get_formula(model, input_dim):
    x = [Atom(i + 1) for i in range(input_dim)]
    all = set()
    for i in x:
        all.add(i)

    for layer in model:
        assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
        if isinstance(layer, GroupSum):  # TODO: make get_formula for GroupSum
            continue
        x = layer.get_formula(x)
        for o in x:
            all.add(o)

    print(x)
