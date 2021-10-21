"""Main script with training and evaluation functions."""
from collections import defaultdict
from torch import optim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fix_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda')
    return device


def evaluate(net, data_loader, loss_fn, epoch, num_epochs, mode="test"):
    """Evalues given model on given dataloader."""

    device = get_device()

    # turn off stuff like drop-out (if it exists)
    net = net.eval()

    y_true = []
    y_pred = []
    loss = 0.0

    iterator = tqdm(
        data_loader,
        f"Evaluate: Epoch [{epoch}/{num_epochs}]", bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
    )
    with torch.no_grad():
        for i, (images, labels) in enumerate(iterator):

            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            batch_loss = loss_fn(outputs, labels).item()
            loss += batch_loss

            # collect predictions
            batch_y_pred = outputs.argmax(1)
            batch_y_true = labels

            if batch_y_pred.device != "cpu":
                batch_y_pred = batch_y_pred.detach()
            if batch_y_true.device != "cpu":
                batch_y_true = batch_y_true.detach()

            y_pred.append(batch_y_pred.cpu().numpy())
            y_true.append(batch_y_true.cpu().numpy())

    # aggregate batch predictions and ground-truth
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    # compute epoch losses and metrics
    loss /= len(data_loader)
    accuracy = np.sum(y_pred == y_true) / len(y_true)

    # log statistics
    print(f'{mode.upper()} \t: Summary: Loss: {loss:.4f} Accuracy: {accuracy:.4f}')

    return loss, accuracy


def train(net, loss_fn, train_loader, test_loader, num_epochs, opt, sch=None):
    """Trains a given network on given train loader."""
    device = get_device()

    batch_losses = defaultdict(list)
    train_epoch_losses = defaultdict(list)
    train_epoch_metrics = defaultdict(list)
    test_epoch_losses = defaultdict(list)
    test_epoch_metrics = defaultdict(list)
    
    epochs = list(range(1, num_epochs + 1))
    for epoch in epochs:

        y_true = []
        y_pred = []

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        iterator = tqdm(
            train_loader,
            f"Training: Epoch [{epoch}/{num_epochs}]", bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
        )
        for i, batch in enumerate(iterator):
            (images, labels) = batch

            images = images.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            outputs = net(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            opt.step()

            # collect losses
            batch_loss = loss.item()
            batch_losses["loss"].append(batch_loss)
            epoch_loss += batch_loss

            # collect predictions
            batch_y_pred = outputs.argmax(1)
            batch_y_true = labels

            if batch_y_pred.device != "cpu":
                batch_y_pred = batch_y_pred.detach()
            if batch_y_true.device != "cpu":
                batch_y_true = batch_y_true.detach()

            y_pred.append(batch_y_pred.cpu().numpy())
            y_true.append(batch_y_true.cpu().numpy())

        # aggregate batch predictions and ground-truth
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        # collect epoch losses
        epoch_loss /= len(train_loader)
        train_epoch_losses["loss"].append(epoch_loss)

        # compute epoch metrics
        epoch_accuracy = np.sum(y_pred == y_true) / len(y_true)
        train_epoch_metrics["accuracy"].append(epoch_accuracy)

        # compute loss and metrics on the test set
        test_loss, test_accuracy = evaluate(net, test_loader, loss_fn, epoch, num_epochs, mode="test")
        test_epoch_losses["loss"].append(test_loss)
        test_epoch_metrics["accuracy"].append(test_accuracy)

        # log epoch statistics
        print(f'TRAIN \t: Summary: Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}')

    return epochs, train_epoch_losses, train_epoch_metrics, test_epoch_losses, test_epoch_metrics


if __name__ == "__main__":
    # import relevant modules
    from data.input_transforms import InputTransform
    from data.cifar import CIFAR
    from data.dataloader import get_dataloader
    from models.optimizer import optimizer, scheduler
    from networks.twolayernet import TwolayerNet
    from utils.viz import plot_multiple_quantities_by_time

    # fix randomness
    fix_seed(0)

    # define the input transforms (incl. augmentations)
    train_transform_list = [
        {
            "name": "ToTensor",
            "args": {},
        },
        {
            "name": "Normalize",
            "args": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        },
    ]
    train_transform = InputTransform(train_transform_list)
    test_transform_list = [
        {
            "name": "ToTensor",
            "args": {},
        },
        {
            "name": "Normalize",
            "args": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        },
    ]
    test_transform = InputTransform(test_transform_list)

    # define the dataset
    train_dataset = CIFAR(root="./datasets/CIFAR-10/", train=True, transform=train_transform)
    test_dataset = CIFAR(root="./datasets/CIFAR-10/", train=False, transform=test_transform)

    # obtain the train dataloader
    train_loader = get_dataloader(train_dataset, train=True, batch_size=32, num_workers=1)
    test_loader = get_dataloader(test_dataset, train=False, batch_size=32, num_workers=1)

    # define the network (arch)
    net = TwolayerNet(num_inputs=3 * 32 * 32, num_hidden=512, num_classes=10)

    # define the optimizer and scheduler
    opt = optimizer(model_params=net.parameters(), name="SGD", lr=1e-3, momentum=0.9)
    sch = scheduler(opt=opt, name="StepLR", step_size=10, gamma=0.1)

    # define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # train the model
    epochs, train_losses, train_metrics, test_losses, test_metrics = train(
        net, loss_fn, train_loader, test_loader, num_epochs=2, opt=opt, sch=sch,
    )

    # plot training curves
    plot_multiple_quantities_by_time(
        quantities=[train_losses["loss"], test_losses["loss"]],
        time=epochs,
        labels=["Train", "Test"],
        title="Loss curves",
        show=False,
        save=True,
        save_path="./results/loss_plot.png",
        ylabel="Loss",
    )
    plot_multiple_quantities_by_time(
        quantities=[train_metrics["accuracy"], test_metrics["accuracy"]],
        time=epochs,
        labels=["Train", "Test"],
        title="Accuracy curves",
        show=False,
        save=True,
        save_path="./results/accu_plot.png",
        ylabel="Accuracy",
    )