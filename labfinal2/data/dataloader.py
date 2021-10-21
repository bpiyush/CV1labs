"""Defines dataloader."""
from torch.utils.data import Dataset, DataLoader


def get_dataloader(
        dataset: Dataset,
        train: bool,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
    """Returns dataloader for given dataset"""
    dataloader = DataLoader(
        dataset, shuffle=train, batch_size=batch_size, num_workers=num_workers,
    )
    return dataloader


if __name__ == "__main__":
    from cifar import CIFAR

    dataset = CIFAR(root="../datasets/CIFAR-10/", train=True)
    dataloader = get_dataloader(dataset, train=True, batch_size=1)
    assert len(dataloader) == len(dataset)
