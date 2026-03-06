from __future__ import annotations

<<<<<<< HEAD
from collections import OrderedDict

import torch


def get_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict()

    for key, value in params_dict:
        tensor = torch.tensor(value, dtype=model.state_dict()[key].dtype)
        state_dict[key] = tensor

    model.load_state_dict(state_dict, strict=True)
=======
from typing import List
import numpy as np
import torch
import torch.nn as nn


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as NumPy arrays (Flower format)."""
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load NumPy parameters into the model (Flower format)."""
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)
>>>>>>> 0a736d8f481586f99da1faeedb3b3e80bfaa5e25
