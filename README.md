# MAS + Decentralized FL (CIFAR-100)

## Run Flower + Decentralized MAS (default)

```powershell
flwr run .
```

This uses `decentralized-mode = true` from `pyproject.toml`.

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
