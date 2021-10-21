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
