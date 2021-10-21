"""Defines transforms for inputs (here, images)."""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class InputTransform:
    def __init__(self, transform_list: list) -> None:
        
        # define a sequence of transforms to be applied
        self._transforms = []
        for t in transform_list:
            assert set(t.keys()) == {"name", "args"}

            T = getattr(transforms, t["name"])(**t["args"])
            self._transforms.append(T)
        
        self._transforms = transforms.Compose(self._transforms)
    
    def __call__(self, img: np.ndarray):
        return self._transforms(img)


if __name__ == "__main__":
    # test basic transforms
    transform_list = [
        {
            "name": "ToTensor",
            "args": {},
        },
        {
            "name": "Normalize",
            "args": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        },
    ]
    IT = InputTransform(transform_list)

    x = np.random.randn(32, 32, 3)
    tx = IT(x)
    assert tx.shape == torch.Size([3, 32, 32])

    # test more transforms (augmentations)
    transform_list = [
        {
            "name": "ToTensor",
            "args": {},
        },
        {
            "name": "Normalize",
            "args": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        },
        {
            "name": "ColorJitter",
            "args": dict(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        },
        {
            "name": "RandomAffine",
            "args": {"degrees": 30, "translate": (0.1, 0.1), "scale": (0.5, 1.5)},
        },
        {
            "name": "RandomHorizontalFlip",
            "args": {"p": 0.5},
        },
        {
            "name": "RandomResizedCrop",
            "args": {"size": 32},
        },
        {
            "name": "GaussianBlur",
            "args": {"kernel_size": 5},
        },
        {
            "name": "RandomErasing",
            "args": {},
        },
    ]
    IT = InputTransform(transform_list)

    x = np.random.randn(32, 32, 3)
    tx = IT(x)
    assert tx.shape == torch.Size([3, 32, 32])


