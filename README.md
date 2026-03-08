# MAS + Decentralized FL (CIFAR-100)

## Project description

This project implements a multi-agent system (MAS) with decentralized federated learning using Flower.
The training process uses CIFAR-100 and is organized around three agent roles:

- `StorageAgent`: loads and partitions data, manages local histories and checkpoints.
- `ComputeAgent`: trains a local model on each node and sends model updates.
- `AggregationAgent`: performs decentralized aggregation logic and updates node/global parameters.

Decentralized learning is implemented through a custom Flower strategy where each node updates using neighbor-based aggregation (ring topology) instead of classic single-step global averaging.
Server-side evaluation is performed after aggregation on the shared CIFAR-100 test split.

## CIFAR-100 implementation details

- Dataset source: Hugging Face `uoft-cs/cifar100`.
- Cache location: `data/hf_cache`.
- Supported partition modes:
  - `iid`: random equal split across clients.
  - `dirichlet` (`non-IID`): class-imbalanced split controlled by `dirichlet-alpha`.
- Train transforms: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, normalization.
- Test transforms: normalization only.

## Run Flower + Decentralized MAS (default)

```powershell
flwr run .
```

This uses `decentralized-mode = true` from `pyproject.toml`.
Data partitioning is controlled by:
- `partition-mode = "iid"` or `"dirichlet"`
- `dirichlet-alpha` (used when `partition-mode="dirichlet"`)

## Run Flower + centralized FedAvg

```powershell
flwr run . --run-config "decentralized-mode=false"
```

## Run standalone decentralized simulation (legacy)

```powershell
python -m src.decentralized_sim
```

or (after install):

```powershell
decentralized-train
```

## Plot metrics

```powershell
python plot_metrics.py
```

Plots are saved to `plots/`.
