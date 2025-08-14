import torch
import numpy as np

from pysat.formula import *

BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


# | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
# |----|----------------------|-------|-------|-------|-------|
# | 0  | 0                    | 0     | 0     | 0     | 0     |
# | 1  | A and B              | 0     | 0     | 0     | 1     |
# | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
# | 3  | A                    | 0     | 0     | 1     | 1     |
# | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
# | 5  | B                    | 0     | 1     | 0     | 1     |
# | 6  | A xor B              | 0     | 1     | 1     | 0     |
# | 7  | A or B               | 0     | 1     | 1     | 1     |
# | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
# | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
# | 10 | not(B)               | 1     | 0     | 1     | 0     |
# | 11 | B implies A          | 1     | 0     | 1     | 1     |
# | 12 | not(A)               | 1     | 1     | 0     | 0     |
# | 13 | A implies B          | 1     | 1     | 0     | 1     |
# | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
# | 15 | 1                    | 1     | 1     | 1     | 1     |


def bit_add(*args):
    return [i for i in args]


def idx_to_formula(a, b, i):
    if i == 0:
        return Atom(False)  # 0
    if i == 1:
        return And(a, b)  # 2
    if i == 2:
        return Neg(Implies(a, b))
    if i == 3:
        return a
    if i == 4:
        return Neg(Implies(b, a))
    if i == 5:
        return b
    if i == 6:
        return Or(And(a, Neg(b)), And(b, Neg(a)))  # 4
    if i == 7:
        return Or(a, b)  # 1
    if i == 8:
        return Neg(Or(a, b))
    if i == 9:
        return Neg(XOr(a, b))
    if i == 10:
        return Neg(b)
    if i == 11:
        return Implies(b, a)
    if i == 12:
        return Neg(a)
    if i == 13:
        return Implies(a, b)
    if i == 14:
        return Neg(And(a, b))
    if i == 15:
        return Atom(True)


def idx_to_op(i):
    if i == 0:
        return "FALSE"
    elif i == 1:
        return "a AND b"
    elif i == 2:
        return "NOT (a IMPLY b)"
    elif i == 3:
        return "a"
    elif i == 4:
        return "NOT (b IMPLY a)"
    elif i == 5:
        return "b"
    elif i == 6:
        return "a XOR b"
    elif i == 7:
        return "a OR b"
    elif i == 8:
        return "not (a OR b)"
    elif i == 9:
        return "not (a XOR b)"
    elif i == 10:
        return "not B"
    elif i == 11:
        return "b IMPLY a"
    elif i == 12:
        return "not A"
    elif i == 13:
        return "a IMPLY b"
    elif i == 14:
        return "not (a AND b)"
    elif i == 15:
        return "TRUE"

    assert False


def bin_op(a, b, i):
    assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
    if a.shape[0] > 1:
        assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)

    if i == 0:
        return torch.zeros_like(a)
    elif i == 1:
        return a * b
    elif i == 2:
        return a - a * b
    elif i == 3:
        return a
    elif i == 4:
        return b - a * b
    elif i == 5:
        return b
    elif i == 6:
        return a + b - 2 * a * b
    elif i == 7:
        return a + b - a * b
    elif i == 8:
        return 1 - (a + b - a * b)
    elif i == 9:
        return 1 - (a + b - 2 * a * b)
    elif i == 10:
        return 1 - b
    elif i == 11:
        return 1 - b + a * b
    elif i == 12:
        return 1 - a
    elif i == 13:
        return 1 - a + a * b
    elif i == 14:
        return 1 - a * b
    elif i == 15:
        return torch.ones_like(a)


def bin_op_s(a, b, i_s):
    # a (batch_size, neuron_number): subset of input features
    # b (batch_size, neuron_number): subset of input features
    # i_s (neuron_number, bin_op_number): weight of bin_op (softmax'ed)

    # r = torch.zeros_like(a)
    # for i in range(16):
    #     u = bin_op(a, b, i)
    #     r = r + i_s[..., i] * u
    # return r

    y = torch.stack([bin_op(a, b, i) for i in range(16)], dim=2)
    y = i_s * y
    y = y.sum(dim=-1)
    return y


########################################################################################################################


def get_unique_connections(in_dim, out_dim, device="cuda"):
    assert out_dim * 2 >= in_dim, (
        "The number of neurons ({}) must not be smaller than half of the number of inputs "
        "({}) because otherwise not all inputs could be used or considered.".format(
            out_dim, in_dim
        )
    )

    x = torch.arange(in_dim).long().unsqueeze(0)

    # Take pairs (0, 1), (2, 3), (4, 5), ...
    a, b = x[..., ::2], x[..., 1::2]
    if a.shape[-1] != b.shape[-1]:
        m = min(a.shape[-1], b.shape[-1])
        a = a[..., :m]
        b = b[..., :m]

    # If this was not enough, take pairs (1, 2), (3, 4), (5, 6), ...
    if a.shape[-1] < out_dim:
        a_, b_ = x[..., 1::2], x[..., 2::2]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        if a.shape[-1] != b.shape[-1]:
            m = min(a.shape[-1], b.shape[-1])
            a = a[..., :m]
            b = b[..., :m]

    # If this was not enough, take pairs with offsets >= 2:
    offset = 2
    while out_dim > a.shape[-1] > offset:
        a_, b_ = x[..., :-offset], x[..., offset:]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        offset += 1
        assert a.shape[-1] == b.shape[-1], (a.shape[-1], b.shape[-1])

    if a.shape[-1] >= out_dim:
        a = a[..., :out_dim]
        b = b[..., :out_dim]
    else:
        assert False, (a.shape[-1], offset, out_dim)

    perm = torch.randperm(out_dim)

    a = a[:, perm].squeeze(0)
    b = b[:, perm].squeeze(0)

    a, b = a.to(torch.int64), b.to(torch.int64)
    a, b = a.to(device), b.to(device)
    a, b = a.contiguous(), b.contiguous()
    return a, b


########################################################################################################################


class GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None


########################################################################################################################
