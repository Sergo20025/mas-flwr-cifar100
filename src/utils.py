from __future__ import annotations

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