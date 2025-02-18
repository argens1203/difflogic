import torch
import torchvision

import mnist_dataset
from difflogic import CompiledLogicNet

torch.set_num_threads(1)

dataset = "cifar-10-3-thresholds"
batch_size = 1_000

transform = {
    "cifar-10-3-thresholds": lambda x: torch.cat(
        [(x > (i + 1) / 4).float() for i in range(3)], dim=0
    ),
    "cifar-10-31-thresholds": lambda x: torch.cat(
        [(x > (i + 1) / 32).float() for i in range(31)], dim=0
    ),
}[dataset]
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(transform),
    ]
)
train_set = torchvision.datasets.CIFAR10(
    "./data-cifar", train=True, download=True, transform=transforms
)
test_set = torchvision.datasets.CIFAR10(
    "./data-cifar", train=False, transform=transforms
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True
)

num_bits = 64
save_lib_path = "lib/00520000_64.so"
print(save_lib_path)
compiled_model = CompiledLogicNet.load(save_lib_path, 10, num_bits)

correct, total = 0, 0
for data, labels in test_loader:
    data = torch.nn.Flatten()(data).bool().numpy()

    output = compiled_model.forward(data, verbose=True)

    correct += (output.argmax(-1) == labels).float().sum()
    total += output.shape[0]

acc3 = correct / total
print("COMPILED MODEL", num_bits, acc3)
