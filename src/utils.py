import os
import random
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def read_cifar(train: bool = True) -> Tuple[np.ndarray, Any]:
    data = []
    targets = []

    if train:
        for idx in range(1, 6):
            entry = unpickle(f"data/data_batch_{idx}")
            data.extend(entry[b"data"])
            targets.extend(entry[b"labels"])
    else: 
        entry = unpickle(f"data/test_batch")
        data.extend(entry[b"data"])
        targets.extend(entry[b"labels"])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC

    return data, targets

def read_cifar_meta() -> Tuple[List, Dict]:
    meta = unpickle("data/batches.meta")
    classes = meta[b"label_names"]
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    return classes, class_to_idx

def normalize(tensor: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False) -> torch.Tensor:
    if not inplace:
        tensor = torch.clone(tensor)

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    return tensor.sub_(mean).div_(std)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)