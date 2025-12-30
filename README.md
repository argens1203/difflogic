# difflogic - A Library for Differentiable Logic Gate Networks

![difflogic_logo](difflogic_logo.png)

> **Note:** This repository is a modified version of the [original difflogic repository](https://github.com/Felix-Petersen/difflogic), extended as part of the Master's thesis *"Efficient Formal Explainability of Logic Gate Networks"* at Monash University.
>
> **Author:** Argens Ng (ng.argens@gmail.com)
> **Supervisor:** Alexey Ignatiev

This repository includes the official implementation of our NeurIPS 2022 Paper "Deep Differentiable Logic Gate Networks"
(Paper @ [ArXiv](https://arxiv.org/abs/2210.08277)).

The goal behind differentiable logic gate networks is to solve machine learning tasks by learning combinations of logic
gates, i.e., so-called logic gate networks. As logic gate networks are conventionally non-differentiable, they can
conventionally not be trained with methods such as gradient descent. Thus, differentiable logic gate networks are a
differentiable relaxation of logic gate networks which allows efficiently learning of logic gate networks with gradient
descent. Specifically, `difflogic` combines real-valued logics and a continuously parameterized relaxation of
the network. This allows learning which logic gate (out of 16 possible) is optimal for each neuron.
The resulting discretized logic gate networks achieve fast inference speeds, e.g., beyond a million images
of MNIST per second on a single CPU core.

`difflogic` is a Python 3.6+ and PyTorch 1.9.0+ based library for training and inference with logic gate networks.
The library can be installed with:
```shell
pip install difflogic
```
> Note that `difflogic` requires CUDA, the CUDA Toolkit (for compilation), and `torch>=1.9.0` (matching the CUDA version).

For additional installation support, see [INSTALLATION_SUPPORT.md](INSTALLATION_SUPPORT.md).

## Table of Contents

- [Intro and Training](#-intro-and-training)
- [Model Inference](#-model-inference)
- [Experiments](#-experiments)
- [LGN Explanation Framework](#-lgn-explanation-framework)
- [Installation Notes](#-installation-notes)
- [Citing](#-citing)
- [License](#-license)

## Intro and Training

This library provides a framework for both training and inference with logic gate networks.
The following gives an example of a definition of a differentiable logic network model for the MNIST data set:

```python
from difflogic import LogicLayer, GroupSum
import torch

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    LogicLayer(784, 16_000),
    LogicLayer(16_000, 16_000),
    LogicLayer(16_000, 16_000),
    LogicLayer(16_000, 16_000),
    LogicLayer(16_000, 16_000),
    GroupSum(k=10, tau=30)
)
```

This model receives a `784` dimensional input and returns `k=10` values corresponding to the 10 classes of MNIST.
The model may be trained, e.g., with a `torch.nn.CrossEntropyLoss` similar to how other neural networks models are trained in PyTorch.
Notably, the Adam optimizer (`torch.optim.Adam`) should be used for training and the recommended default learning rate is `0.01` instead of `0.001`.
Finally, it is also important to note that the number of neurons in each layer is much higher for logic gate networks compared to
conventional MLP neural networks because logic gate networks are very sparse.

To go into details, for each of these modules, in the following we provide more in-depth examples:

```python
layer = LogicLayer(
    in_dim=784,             # number of inputs
    out_dim=16_000,         # number of outputs
    device='cuda',          # the device (cuda / cpu)
    implementation='cuda',  # the implementation to be used (native cuda / vanilla pytorch)
    connections='random',   # the method for the random initialization of the connections
    grad_factor=1.1,        # for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
)
```

At this point, it is important to discuss the options for `device` and the provided implementations. Specifically,
`difflogic` provides two implementations (both of which work with PyTorch):

* **`python`** the Python implementation is a substantially slower implementation that is easy to understand as it is implemented directly in Python with PyTorch and does not require any C++ / CUDA extensions. It is compatible with `device='cpu'` and `device='cuda'`.
* **`cuda`** is a well-optimized implementation that runs natively on CUDA via custom extensions. This implementation is around 50 to 100 times faster than the python implementation (for large models). It only supports `device='cuda'`.

To aggregate output neurons into a lower dimensional output space, we can use `GroupSum`, which aggregates a number of output neurons into
a `k` dimensional output, e.g., `k=10` for a 10-dimensional classification setting.
It is important to set the parameter `tau`, which the sum of neurons is divided by to keep the range reasonable.
As each neuron has a value between 0 and 1 (or in inference a value of 0 or 1), assuming `n` output neurons of the last `LogicLayer`,
the range of outputs is `[0, n / k / tau]`.

## Model Inference

During training, the model should remain in the PyTorch training mode (`.train()`), which keeps the model differentiable.
However, we can easily switch the model to a hard / discrete / non-differentiable model by calling `model.eval()`, i.e., for inference.
Typically, this will simply discretize the model but not make it faster per se.

However, there are two modes that allow for fast inference:

### `PackBitsTensor`

The first option is to use a `PackBitsTensor`.
`PackBitsTensor`s allow efficient dynamic execution of trained logic gate networks on GPU.

A `PackBitsTensor` can package a tensor (of shape `b x n`) with boolean
data type in a way such that each boolean entry requires only a single bit (in contrast to the full byte typically
required by a bool) by packing the bits along the batch dimension. If we choose to pack the bits into the `int32` data
type (the options are 8, 16, 32, and 64 bits), we would receive a tensor of shape `ceil(b/32) x n` of dtype `int32`.
To create a `PackBitsTensor` from a boolean tensor `data`, simply call:
```python
data_bits = difflogic.PackBitsTensor(data)
```
To apply a model to the `PackBitsTensor`, simply call:
```python
output = model(data_bits)
```
This requires that the `model` is in `.eval()` mode, and if supplied with a `PackBitsTensor`, will automatically use
a logic gate-based inference on the tensor. This also requires that `model.implementation = 'cuda'` as the mode is only
implemented in CUDA.
It is notable that, while the model is in `.eval()` mode, we can still also feed float tensors through the model, in
which case it will simply use a hard variant of the real-valued logics.

### `CompiledLogicNet`

The second option is to use a `CompiledLogicNet`.
This allows especially efficient static execution of a fixed trained logic gate network on CPU.
Specifically, `CompiledLogicNet` converts a model into efficient C code and can compile this code into a binary that
can then be efficiently run or exported for applications.
The following is an example for creating `CompiledLogicNet` from a trained `model`:

```python
compiled_model = difflogic.CompiledLogicNet(
    model=model,            # the trained model (should be a `torch.nn.Sequential` with `LogicLayer`s)
    num_bits=64,            # the number of bits of the datatype used for inference (typically 64 is fastest, should not be larger than batch size)
    cpu_compiler='gcc',     # the compiler to use for the c code (alternative: clang)
    verbose=True
)
compiled_model.compile(
    save_lib_path='my_model_binary.so',  # the (optional) location for storing the binary such that it can be reused
    verbose=True
)

# to apply the model, we need a 2d numpy array of dtype bool, e.g., via  `data = data.bool().numpy()`
output = compiled_model(data)
```

This will compile a model into a shared object binary, which is then automatically imported.
To export this to other applications, one may either call the shared object binary from another program or export
the model into C code via `compiled_model.get_c_code()`.
A limitation of the current `CompiledLogicNet` is that the compilation time can become long for large models.

We note that between publishing the paper and the publication of `difflogic`, we have substantially improved the implementations.
Thus, the model inference modes have some deviation from the implementations for the original paper as we have
focussed on making it more scalable, efficient, and easier to apply in applications.
We have especially focussed on modularity and efficiency for larger models and have opted to polish the presented
implementations over publishing a plethora of different competing implementations.

## Experiments

In the following, we present a few example experiments which are contained in the `experiments` directory.
`main.py` executes the experiments for difflogic and `main_baseline.py` contains regular neural network baselines.

### Adult / Breast Cancer

```shell
python experiments/main.py  -eid 526010           -bs 100 -t 20 --dataset adult         -ni 100_000 -ef 1_000 -k 256 -l 5 --compile_model
python experiments/main.py  -eid 526020 -lr 0.001 -bs 100 -t 20 --dataset breast_cancer -ni 100_000 -ef 1_000 -k 128 -l 5 --compile_model
```

### MNIST

```shell
python experiments/main.py  -bs 100 -t  10 --dataset mnist20x20 -ni 200_000 -ef 1_000 -k  8_000 -l 6 --compile_model
python experiments/main.py  -bs 100 -t  30 --dataset mnist      -ni 200_000 -ef 1_000 -k 64_000 -l 6 --compile_model
# Baselines:
python experiments/main_baseline.py  -bs 100 --dataset mnist    -ni 200_000 -ef 1_000 -k  128 -l 3
python experiments/main_baseline.py  -bs 100 --dataset mnist    -ni 200_000 -ef 1_000 -k 2048 -l 7
```

### CIFAR-10

```shell
python experiments/main.py  -bs 100 -t 100 --dataset cifar-10-3-thresholds  -ni 200_000 -ef 1_000 -k    12_000 -l 4 --compile_model
python experiments/main.py  -bs 100 -t 100 --dataset cifar-10-3-thresholds  -ni 200_000 -ef 1_000 -k   128_000 -l 4 --compile_model
python experiments/main.py  -bs 100 -t 100 --dataset cifar-10-31-thresholds -ni 200_000 -ef 1_000 -k   256_000 -l 5
python experiments/main.py  -bs 100 -t 100 --dataset cifar-10-31-thresholds -ni 200_000 -ef 1_000 -k   512_000 -l 5
python experiments/main.py  -bs 100 -t 100 --dataset cifar-10-31-thresholds -ni 200_000 -ef 1_000 -k 1_024_000 -l 5
```

## LGN Explanation Framework

This repository also includes the Logic Gate Network (LGN) explanation framework, which converts trained models to SAT formulas and generates explanations using SAT solvers.

### Supported Datasets

| Dataset | Description |
|---------|-------------|
| `iris` | Iris flower classification (4 features, 3 classes) |
| `monk1`, `monk2`, `monk3` | MONK's Problems (6 features, 2 classes) |
| `breast_cancer` | Wisconsin Breast Cancer (9 features, 2 classes) |
| `adult` | Adult Income dataset (14 features, 2 classes) |
| `compas` | COMPAS Recidivism dataset |
| `lending` | Lending Club dataset |
| `mnist` | MNIST handwritten digits (784 features, 10 classes) |

### Command-Line Arguments

#### Model Parameters

| Argument | Description |
|----------|-------------|
| `-k`, `--num_neurons` | Number of neurons per layer |
| `-l`, `--num_layers` | Number of logic layers |
| `-ni`, `--num_iterations` | Number of training iterations |
| `-ef`, `--eval_freq` | Evaluation frequency during training |
| `-bs`, `--batch_size` | Training batch size |
| `-t`, `--tau` | Temperature parameter for GroupSum |
| `-lr`, `--learning_rate` | Learning rate (default: 0.01) |
| `--grad_factor` | Gradient factor for deep networks (default: 1.0) |

#### Model I/O

| Argument | Description |
|----------|-------------|
| `--save_model` | Save trained model to disk |
| `--load_model` | Load pre-trained model from disk |
| `--compile_model` | Compile model to C code for fast inference |

#### Explanation Options

| Argument | Description |
|----------|-------------|
| `--explain=<values>` | Explain a specific instance (comma-separated feature values) |
| `--explain_one` | Explain a single test instance |
| `--explain_all` | Explain all test instances |
| `--xnum=<n>` | Maximum number of explanations to generate |

#### SAT Encoding Options

| Argument | Values | Description |
|----------|--------|-------------|
| `--deduplicate` | `sat`, `bdd`, `None` | Deduplication strategy for SAT encoding |
| `--enc_type_at_least` | `tot`, `mtot`, `kmtot`, `native` | Encoding type for at-least constraints |
| `--enc_type_eq` | `bit`, `tot`, `native` | Encoding type for equality constraints |

#### Explanation Algorithm Options

| Argument | Values | Description |
|----------|--------|-------------|
| `--explain_algorithm` | `mus`, `mcs`, `both`, `find_one`, `var` | Explanation enumeration algorithm |
| `--h_type` | `sorted`, `lbx`, `sat` | Hitman oracle type |
| `--h_solver` | `mgh`, `cd195`, `g3` | Hitman SAT solver |
| `--solver_type` | `gc3`, `g3`, `cd195` | Main SAT solver |

### Usage Examples

#### First-Time Setup: Train and Save a Model

```shell
python main.py --dataset iris -bs 100 -ni 2000 -ef 1000 -k 6 -l 2 --save_model
```

#### Load Model and Extract Logic Gates

```shell
python main.py --dataset iris --load_model --get_formula
```

#### Explain a Single Instance

```shell
# Iris dataset (sepal_length, sepal_width, petal_length, petal_width)
python main.py --dataset iris --load_model --explain=5.1,3.3,1.7,0.5

# Adult dataset
python main.py --dataset adult --load_model --explain=17,Private,11th,Never-married,Sales,Own-child,White,Female,0,0,10,United-States

# MONK datasets
python main.py --dataset monk1 --load_model --explain=1,1,1,1,4,1

# Breast Cancer dataset
python main.py --dataset breast_cancer --load_model --explain=60-69,ge40,20-24,0-2,no,3,right,left_low,no
```

#### Explain with Different Encoding Types

```shell
python main.py --dataset iris --load_model --explain=5.1,3.3,1.7,0.5 --enc_type_at_least=tot
```

#### Explain One Test Instance with Verbose Output

```shell
python main.py --dataset mnist --load_model --explain_one --xnum=1 --verbose
```

#### Compare Deduplication Strategies

```shell
# Without deduplication
python main.py --dataset iris --explain_one --save_model

# With SAT-based deduplication
python main.py --dataset iris --explain_one --load_model --deduplicate=sat

# With BDD-based deduplication
python main.py --dataset iris --explain_one --load_model --deduplicate=bdd
```

### Running Tests

```shell
cd experiments && python -m unittest
```

## Installation Notes

### CUDA Setup

1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

2. Set the CUDA architecture for your GPU:
   ```shell
   export TORCH_CUDA_ARCH_LIST="8.9"  # Adjust for your GPU
   ```

3. Install Python development headers:
   ```shell
   # Ubuntu/Debian
   sudo apt-get install python3-dev

   # macOS (for CPU-only mode)
   # No additional headers needed
   ```

4. Install the package:
   ```shell
   python setup.py install
   ```

### Graphviz (for BDD visualization)

BDD visualization requires Graphviz:

```shell
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz
```

### CPU-Only Installation (macOS)

For CPU-only usage without CUDA:

1. Comment out `ext_modules` in `setup.py`
2. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```

## Citing

```bibtex
@inproceedings{petersen2022difflogic,
  title={{Deep Differentiable Logic Gate Networks}},
  author={Petersen, Felix and Borgelt, Christian and Kuehne, Hilde and Deussen, Oliver},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

## License

`difflogic` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.

Patent pending.
