import torch
from attr import dataclass

from difflogic import LogicLayer, GroupSum

from lgn.dataset import get_dataset
from constant import device


def get_model(args, results=None):
    llkw = dict(grad_factor=args.grad_factor, connections=args.connections)
    dataset = get_dataset(args.dataset)
    in_dim = dataset.get_input_dim()
    class_count = dataset.get_num_of_classes()

    logic_layers = []

    k = args.num_neurons
    l = args.num_layers

    ####################################################################################################################

    logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
    for _ in range(l - 1):
        logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

    model = torch.nn.Sequential(*logic_layers, GroupSum(class_count, args.tau))

    ####################################################################################################################

    total_num_neurons = sum(
        map(lambda x: x.num_neurons, logic_layers[:])
    )  # TODO: Why 1:-1?
    # if args.verbose in ["debug", "info"]:
    # print(f"total_num_neurons={total_num_neurons}")
    total_num_weights = (
        sum(map(lambda x: x.num_weights, logic_layers[:])) * 16
    )  # TODO: Why 1:-1?
    # if args.verbose in ["debug", "info"]:
    # print(f"total_num_weights={total_num_weights}")
    if results is not None:
        results.store_results(
            {
                "total_num_neurons": total_num_neurons,
                "total_num_weights": total_num_weights,
            }
        )

    model = model.to(device)

    # print(model)
    # input("Press Enter to continue...")
    if results is not None:
        results.store_results({"model_str": str(model)})

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer


import os
from difflogic import CompiledLogicNet


def compile_model(args, model, test_loader):
    print("\n" + "=" * 80)
    print(" Converting the model to C code and compiling it...")
    print("=" * 80)

    for opt_level in range(4):

        for num_bits in [
            # 8,
            # 16,
            # 32,
            64
        ]:
            os.makedirs("lib", exist_ok=True)
            save_lib_path = "lib/{:08d}_{}.so".format(
                args.experiment_id if args.experiment_id is not None else 0,
                num_bits,
            )

            compiled_model = CompiledLogicNet(
                model=model,
                num_bits=num_bits,
                cpu_compiler="gcc",
                # cpu_compiler='clang',
                verbose=True,
            )

            compiled_model.compile(
                opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
                save_lib_path=save_lib_path,
                verbose=args.verbose in ["debug", "info"],
            )

            correct, total = 0, 0
            with torch.no_grad():
                for data, labels in torch.utils.data.DataLoader(
                    test_loader.dataset, batch_size=int(1e6), shuffle=False
                ):
                    data = torch.nn.Flatten()(data).bool().numpy()

                    output = compiled_model(data, verbose=True)

                    correct += (output.argmax(-1) == labels).float().sum()
                    total += output.shape[0]

            acc3 = correct / total
            print("COMPILED MODEL", num_bits, acc3)
