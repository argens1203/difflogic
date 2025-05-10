import torch

# import difflogic_cuda
import numpy as np
from .functional import (
    bin_op_s,
    get_unique_connections,
    GradFactor,
    idx_to_op,
    idx_to_formula,
    bit_add,
)

from .packbitstensor import PackBitsTensor


########################################################################################################################


class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        ),
        grad_factor: float = 1.0,
        implementation: str = None,
        connections: str = "random",
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()
        self.weights = torch.nn.parameter.Parameter(
            torch.randn(out_dim, 16, device=device)
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = implementation
        if self.implementation is None and device == "cuda":
            self.implementation = "cuda"
        elif self.implementation is None and (device == "cpu" or device == "mps"):
            self.implementation = "python"
        assert self.implementation in ["cuda", "python"], self.implementation

        self.connections = connections
        assert self.connections in ["random", "unique"], self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == "cuda":
            """
            Defining additional indices for improving the efficiency of the backward of the CUDA implementation.
            """
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(),
                device=device,
                dtype=torch.int64,
            )
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist],
                dtype=torch.int64,
                device=device,
            )
        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert (
                not self.training
            ), "PackBitsTensor is not supported for the differentiable training mode."
            assert self.device == "cuda", (
                "PackBitsTensor is only supported for CUDA, not for {}. "
                "If you want fast inference on CPU, please use CompiledDiffLogicModel."
                "".format(self.device)
            )

        else:
            if self.grad_factor != 1.0:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == "cuda":
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == "python":
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        # x.shape: batch_size, input_dim
        # self.indices: Tuple (neuron_size, neuron_size)
        # self.weights: (neuron_size, bin_op_number)
        assert (
            x.shape[-1] == self.in_dim
        ), "Input shape {} does not match model.in_dim {}".format(
            x[0].shape[-1], self.in_dim
        )
        self.indices = self.indices[0].long(), self.indices[1].long()

        # Selects 2 subsets of input features as defined in self.get_connection
        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            # weights: (neuron_size, 16)
            # 16 being number of binary operations
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(
                torch.float32
            )
            x = bin_op_s(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == "cuda", x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.training:
            w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x,
                    a,
                    b,
                    w,
                    self.given_x_indices_of_y_start,
                    self.given_x_indices_of_y,
                ).transpose(0, 1)

    def forward_cuda_eval(self, x):
        """
        WARNING: this is an in-place operation.

        :param x:
        :return:
        """
        assert not self.training
        assert isinstance(x, PackBitsTensor)
        assert x.t.shape[0] == self.in_dim, (x.t.shape, self.in_dim)

        a, b = self.indices
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)

        return x

    def extra_repr(self):
        return "{}, {}, {}".format(
            self.in_dim, self.out_dim, "train" if self.training else "eval"
        )

    def get_connections(
        self, connections, device="cuda"
    ):  # Default connnections (is unique, from command line args)
        assert self.out_dim * 2 >= self.in_dim, (
            "The number of neurons ({}) must not be smaller than half of the "
            "number of inputs ({}) because otherwise not all inputs could be "
            "used or considered.".format(self.out_dim, self.in_dim)
        )
        # REMARK: neither subset is complete (which is fine) nor non-repeating (which is also fine)
        # The 2 subsets will become the "left" input and "right" input of all neurons in order
        # Hence subset size = [neuron_size]
        if connections == "random":
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            # c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == "unique":
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)

    def get_formula(self, x):
        # weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(
        # torch.float32
        # repr = [
        #     f"{idx_to_op(i)}: {a.item()}, {b.item()}"  # OP: left_index, right_index
        #     for i, a, b in zip(self.weights.argmax(-1), *self.indices)
        # ]
        # return str(repr)
        formulas = [
            idx_to_formula(x[a], x[b], i)
            for i, a, b in zip(self.weights.argmax(-1), *self.indices)
        ]
        return formulas


########################################################################################################################


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """

    def __init__(self, k: int, tau: float = 1.0, device="cuda"):
        # tau default is 10 from args
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)

        # x.reshape(batch_size, output_dim, input_dim // output_dim) then sum in last dimension
        # ie. summing every X outputs from previous layer, where X is input_dim divided by output_dim
        return (
            x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau
            if self.training
            else x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1)
        )

    def extra_repr(self):
        return "k={}, tau={}".format(self.k, self.tau)

    def get_formula(self, x):
        step_size = len(x) // self.k
        formulas = [bit_add(*[x[i * step_size + j] for j in step_size]) for i in self.k]
        return formulas


########################################################################################################################


class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(
            x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y
        )
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(
                x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y
            )
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None, None


########################################################################################################################
