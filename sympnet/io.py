import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import SympNet

_NAME_TO_ACT = {
    "sigmoid": F.sigmoid,
}
_ACT_TO_NAME = {v: k for k, v in _NAME_TO_ACT.items()}


def save_model(model, path):
    path = Path(path)

    # Ensure that output directory (if any) exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    torch.save(model.state_dict(), path)

    # Save model parameters
    args_path = path.with_suffix(".args")
    args = {
        **model._args,
        "activation": _ACT_TO_NAME[model._args["activation"]],
    }
    with open(args_path, "wt", encoding="utf-8") as fp:
        json.dump(args, fp, indent=2)


def load_model(path):
    path = Path(path)
    
    # Load parameters
    args_path = path.with_suffix(".args")
    with open(args_path, "rt", encoding="utf-8") as fp:
        args = json.load(fp)
    args["activation"] = _NAME_TO_ACT[args["activation"]]

    # Instantiate and load model weights
    model = SympNet(**args)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model
