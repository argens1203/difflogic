import torch
import numpy as np
from tqdm import tqdm

from difflogic import PackBitsTensor

from constant import device, BITS_TO_TORCH_FLOATING_POINT_TYPE
from lgn.dataset.dataset import load_n


def train(model, x, y, loss_fn, optimizer):
    x = model(x)
    loss = loss_fn(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval(model, loader, mode):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        res = np.mean(
            [
                (model(x.float().to(device).round()).argmax(-1) == y.float().to(device))
                .to(torch.float32)
                .mean()
                .item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()


def packbits_eval(model, loader):
    orig_mode = model.training
    with torch.no_grad():
        model.eval()
        res = np.mean(
            [
                (
                    model(
                        PackBitsTensor(
                            x.float().to("cuda").reshape(x.shape[0], -1).round().bool()
                        )
                    ).argmax(-1)
                    == y.float().to("cuda")
                )
                .to(torch.float32)
                .mean()
                .item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()


def multi_eval(
    model,
    train_loader,
    validation_loader,
    test_loader,
    extensive_eval=True,
    results=None,
    packbits_eval=False,
):
    best_acc = 0

    if extensive_eval:
        # Use train mode to test on training data
        train_accuracy_train_mode = eval(model, train_loader, mode=True)
        valid_accuracy_eval_mode = eval(model, validation_loader, mode=False)
        valid_accuracy_train_mode = eval(model, validation_loader, mode=True)
    else:
        train_accuracy_train_mode = -1
        valid_accuracy_eval_mode = -1
        valid_accuracy_train_mode = -1
    # Use evaluation mode to test on training data
    train_accuracy_eval_mode = eval(model, train_loader, mode=False)
    test_accuracy_eval_mode = eval(model, test_loader, mode=False)
    test_accuracy_train_mode = eval(model, test_loader, mode=True)

    r = {
        "train_acc_eval_mode": train_accuracy_eval_mode,
        "train_acc_train_mode": train_accuracy_train_mode,
        "valid_acc_eval_mode": valid_accuracy_eval_mode,
        "valid_acc_train_mode": valid_accuracy_train_mode,
        "test_acc_eval_mode": test_accuracy_eval_mode,
        "test_acc_train_mode": test_accuracy_train_mode,
    }

    if packbits_eval:
        pass
        r["train_acc_eval"] = packbits_eval(model, train_loader)
        r["valid_acc_eval"] = packbits_eval(model, train_loader)
        r["test_acc_eval"] = packbits_eval(model, test_loader)

    if results is not None:
        results.store_results(r)
    else:
        print(r)

    if valid_accuracy_eval_mode > best_acc:
        best_acc = valid_accuracy_eval_mode
        if results is not None:
            results.store_final_results(r)
        else:
            print("IS THE BEST UNTIL NOW.")

    if results is not None:
        results.save()


def train_eval(
    args,
    train_loader,
    validation_loader,
    test_loader,
    model,
    loss_fn,
    optim,
    results=None,
):
    for i, (x, y) in tqdm(
        enumerate(load_n(train_loader, args.num_iterations)),
        desc="iteration",
        total=args.num_iterations,
    ):
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(device)
        y = y.to(device)

        loss = train(model, x, y, loss_fn, optim)

        if (i + 1) % args.eval_freq == 0:
            multi_eval(
                model,
                train_loader,
                validation_loader,
                test_loader,
                results=results,
                packbits_eval=args.packbits_eval,
                extensive_eval=args.extensive_eval,
            )
