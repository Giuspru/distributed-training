import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist  
from s3torchconnector import S3MapDataset, S3ClientConfig
import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
import boto3
from ray.train.torch import prepare_model, prepare_data_loader # WIP preparare il dataset ad una distribuzione tra workers
from torch.utils.data.distributed import DistributedSampler # WIP dataset fino ad ora non faceva effettivamente distribuzione 

# ───────── S3 + shard‐reassembly helpers ────────────────────────────────────


S3_PREFIX = "s3://datasets/mninst" 
ENDPOINT  = os.getenv("AWS_ENDPOINT_URL") #<-- NONE
REGION    = os.getenv("AWS_REGION", "us-east-1")

s3_config = S3ClientConfig(force_path_style=True)


def _download_and_concat(prefix: str) -> bytes:
    """List everything under `prefix`, drop xl.meta, sort the part files, and concat them."""

    ds = S3MapDataset.from_prefix(
        prefix,
        region=REGION,
        endpoint=ENDPOINT,
        transform=None,
        s3client_config=s3_config
    )
    pairs = ds._dataset_bucket_key_pairs
    indices = sorted(
      [i for i, entry in enumerate(pairs) if not entry[1].endswith("xl.meta")],
      key=lambda i: pairs[i][1]
    )

    buf = bytearray()
    for i in indices:
        reader = ds._get_object(i)      
        buf.extend(reader.read())        
    return bytes(buf)

def _parse_idx_images(bytestream: bytes) -> np.ndarray:
    magic, n, rows, cols = struct.unpack_from(">IIII", bytestream, 0)
    data = np.frombuffer(bytestream, dtype=np.uint8, offset=16)
    return data.reshape(n, rows, cols)

def _parse_idx_labels(bytestream: bytes) -> np.ndarray:
    magic, n = struct.unpack_from(">II", bytestream, 0)
    data = np.frombuffer(bytestream, dtype=np.uint8, offset=8)
    return data

def make_dataset(split: str) -> TensorDataset:
    """
    Reassemble and parse the MNIST IDX files for the given split.
    split must be 'train' or 't10k'.
    """
    # 1) Download & stitch images
    img_prefix = f"{S3_PREFIX}/{split}/{'train-images-idx3-ubyte' if split=='train' else 't10k-images-idx3-ubyte'}"
    imgs_bytes = _download_and_concat(img_prefix)

    # 2) Download & stitch labels
    lbl_prefix = f"{S3_PREFIX}/{split}/{'train-labels-idx1-ubyte'  if split=='train' else 't10k-labels-idx1-ubyte'}"
    labs_bytes = _download_and_concat(lbl_prefix)

    # 3) Parse into numpy
    imgs_np = _parse_idx_images(imgs_bytes)       # shape (N,28,28)
    labs_np = _parse_idx_labels(labs_bytes)       # shape (N,)

    # 4) To torch Tensors & normalize
    imgs = torch.from_numpy(imgs_np).unsqueeze(1).float().div(255.0)
    imgs = imgs.sub_(0.1307).div_(0.3081)         # same MNIST normalization
    labs = torch.from_numpy(labs_np).long()

    n = imgs.shape[0]
    if n == 0:
        raise RuntimeError(f"No data found for split={split}")

    print(f"→ Loaded {n} samples from {split}")
    return TensorDataset(imgs, labs)


# ───────── training loop ────────────────────────────────────────────────────

def train_loop_per_worker(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds     = make_dataset("train")
    sampler = DistributedSampler(ds) # WIP Splitting del dataloader per worker
    print("Number of workers: ", (os.getenv("RAY_NUM_WORKERS", 2)))
    loader = DataLoader(ds, batch_size=cfg["bs"], sampler=sampler, shuffle=False, num_workers=int(os.getenv("RAY_NUM_WORKERS", 2))) # <-- nm_workers=4 is a constant, maybe here from the config? cambio shuffle = False
    loader = prepare_data_loader(loader) # WIP

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
    ).to(device)

    model = prepare_model(model) #WIP
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn   = nn.CrossEntropyLoss()

    for epoch in range(cfg["epochs"]):
        sampler.set_epoch(epoch) # WIP <-- questo serve per lo shuffling
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss_fn(model(imgs), labels).backward()
            optimizer.step()

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        os.makedirs("/tmp/model", exist_ok=True)
        model_path = "/tmp/model/mnist.pt"
        torch.save(model.state_dict(), model_path)
        print(f"✔ Model saved to {model_path}")

        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        # WIP: put a timestamp on the model name
        s3_client.upload_file(model_path, "models", "mnist/mnist.pt")
        print("✔ Uploaded model to s3://models/mnist.pt")

# ───────── entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":

    # get epochs and batch size from arguments (the command is python /app/train_mnist_ray.py --epochs 5 --batch-size 16)
    import argparse
    parser = argparse.ArgumentParser(description="Train MNIST model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    

    epochs = args.epochs
    batch_size = args.batch_size

    print(f"Training with {epochs} epochs and batch size {batch_size}")

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={"bs": batch_size, "lr": 1e-3, "epochs": epochs},
        scaling_config=ScalingConfig(
            num_workers=int(os.getenv("RAY_NUM_WORKERS", 2)), 
            use_gpu=torch.cuda.is_available(),
        )
    )
    result = trainer.fit()
    print(result)