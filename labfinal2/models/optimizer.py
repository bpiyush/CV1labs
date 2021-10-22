"""Defines optimizers and schedulers for gradient descent."""
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR


def optimizer(name:str, model_params: dict, **kwargs):
    """Defines an optimizer."""
    assert name in ["SGD", "Adam", "AdamW"]
    opt = eval(name)(params=model_params, **kwargs)
    return opt


def scheduler(opt, name: str, **kwargs):
    """Defines LR scheduler."""
    assert name in ["StepLR", "MultiStepLR", "ExponentialLR"]
    sch = eval(name)(opt, **kwargs)
    return sch
