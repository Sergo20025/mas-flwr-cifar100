"""Построение гистограмм распределения классов для IID и non-IID."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import get_partition_class_counts, get_partition_indices


def _plot_partition_histograms(
    class_counts: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    num_clients, num_classes = class_counts.shape
    cols = 2
    rows = int(np.ceil(num_clients / cols))
    classes = np.arange(num_classes)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 3 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for client_id in range(rows * cols):
        ax = axes.flat[client_id]
        if client_id >= num_clients:
            ax.axis("off")
            continue

        ax.bar(classes, class_counts[client_id], width=0.9)
        ax.set_title(f"Client {client_id}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Samples")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_client_sizes(
    split_indices: list[list[int]],
    title: str,
    out_path: Path,
) -> None:
    client_ids = np.arange(len(split_indices))
    client_sizes = [len(indices) for indices in split_indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(client_ids, client_sizes, width=0.8)
    ax.set_title(title)
    ax.set_xlabel("client_id")
    ax.set_ylabel("num_samples")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iid_counts = get_partition_class_counts(
        num_clients=args.num_clients,
        seed=args.seed,
        partition_mode="iid",
        dirichlet_alpha=args.dirichlet_alpha,
    )
    non_iid_counts = get_partition_class_counts(
        num_clients=args.num_clients,
        seed=args.seed,
        partition_mode="dirichlet",
        dirichlet_alpha=args.dirichlet_alpha,
    )
    iid_indices = get_partition_indices(
        num_clients=args.num_clients,
        seed=args.seed,
        partition_mode="iid",
        dirichlet_alpha=args.dirichlet_alpha,
    )
    non_iid_indices = get_partition_indices(
        num_clients=args.num_clients,
        seed=args.seed,
        partition_mode="dirichlet",
        dirichlet_alpha=args.dirichlet_alpha,
    )

    _plot_partition_histograms(
        iid_counts,
        title="IID Class Distribution by Client",
        out_path=output_dir / "iid_class_distribution.png",
    )
    _plot_partition_histograms(
        non_iid_counts,
        title=f"Non-IID Class Distribution by Client (alpha={args.dirichlet_alpha})",
        out_path=output_dir / "non_iid_class_distribution.png",
    )
    _plot_client_sizes(
        iid_indices,
        title="Client Dataset Sizes (IID Split)",
        out_path=output_dir / "iid_client_dataset_sizes.png",
    )
    _plot_client_sizes(
        non_iid_indices,
        title=f"Client Dataset Sizes (Non-IID Split, alpha={args.dirichlet_alpha})",
        out_path=output_dir / "non_iid_client_dataset_sizes.png",
    )

    print(f"Class distribution plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
