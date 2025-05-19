import numpy as np
import struct
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

def _parse_idx_images(bytestream: bytes) -> np.ndarray:
    magic, n, rows, cols = struct.unpack_from(">IIII", bytestream, 0)
    data = np.frombuffer(bytestream, dtype=np.uint8, offset=16)
    return data.reshape(n, rows, cols)

def _parse_idx_labels(bytestream: bytes) -> np.ndarray:
    magic, n = struct.unpack_from(">II", bytestream, 0)
    data = np.frombuffer(bytestream, dtype=np.uint8, offset=8)
    return data


def test_model_local(data_dir=""):
    """
    Load and test the model using MNIST test data from local files.
    """
    import os

    # Paths to local IDX files
    img_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    lbl_path = os.path.join(data_dir, "t10k-labels.idx1-ubyte")

    # Read local files into memory
    with open(img_path, "rb") as f:
        imgs_bytes = f.read()
    with open(lbl_path, "rb") as f:
        labs_bytes = f.read()

    # Parse IDX data into tensors
    imgs_np = _parse_idx_images(imgs_bytes)
    labs_np = _parse_idx_labels(labs_bytes)

    imgs = torch.from_numpy(imgs_np.copy()).unsqueeze(1).float().div(255.0)
    imgs = imgs.sub_(0.1307).div_(0.3081)  # MNIST normalization
    labs = torch.from_numpy(labs_np).long()
    ds = TensorDataset(imgs, labs)

    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    # Rebuild the model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 10)
    )
    state_dict = torch.load("mnist.pt")
    # Rimuove 'module.' dai nomi dei parametri
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # rimuove il prefisso
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # Run inference
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ Test accuracy: {acc:.2%}")

test_model_local()
