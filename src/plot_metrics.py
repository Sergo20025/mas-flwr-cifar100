"""Построение графиков и boxplot из истории обучения."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _plot_series(rounds, values, ylabel, title, out_path, label: str | None = None):
    plt.figure(figsize=(8, 5))
    if label is None:
        plt.plot(rounds, values, marker="o")
    else:
        plt.plot(rounds, values, marker="o", label=label)
        plt.legend()
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_train_test_comparison(
    train_data,
    test_data,
    metric_key: str,
    ylabel: str,
    title: str,
    out_path,
):
    train_rounds = [float(item["round"]) for item in train_data]
    train_values = [float(item[metric_key]) for item in train_data]
    test_rounds = [float(item["round"]) for item in test_data]
    test_values = [float(item[metric_key]) for item in test_data]

    plt.figure(figsize=(8, 5))
    plt.plot(train_rounds, train_values, marker="o", label="Train")
    plt.plot(test_rounds, test_values, marker="o", label="Test")
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_train_loss_boxplot(fit_data, out_path):
    distributions = []
    labels = []

    for item in fit_data:
        values = item.get("train_loss_distribution")
        if isinstance(values, list) and values:
            distributions.append([float(value) for value in values])
            labels.append(str(int(float(item["round"]))))

    if not distributions:
        return

    plt.figure(figsize=(12, 6))
    plt.boxplot(distributions, tick_labels=labels, showfliers=False)
    plt.xlabel("Round")
    plt.ylabel("Train Loss")
    plt.title("Train Loss Distribution by Round")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _load_history(history_path: Path) -> dict:
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        default="runs/history.json",
        help="Путь к history.json",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Папка для сохранения графиков",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        raise FileNotFoundError(f"{history_path} not found")

    history = _load_history(history_path)
    fit_data = history.get("fit_metrics", []) or history.get("decentralized_fit", [])
    train_eval = history.get("server_evaluate_train", [])
    test_eval = history.get("server_evaluate_test", []) or history.get("server_evaluate", [])

    if not fit_data and not test_eval:
        raise ValueError("No metrics found in history.json")

    plots_dir = Path(args.output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if test_eval:
        rounds = [float(item["round"]) for item in test_eval]
        losses = [float(item["loss"]) for item in test_eval]
        accs = [float(item["accuracy"]) for item in test_eval]
        f1_macro = [float(item["f1_macro"]) for item in test_eval]

        _plot_series(
            rounds,
            losses,
            "Loss",
            "Server Test Loss by Round",
            plots_dir / "loss.png",
        )
        _plot_series(
            rounds,
            accs,
            "Accuracy",
            "Server Test Accuracy by Round",
            plots_dir / "accuracy.png",
        )
        _plot_series(
            rounds,
            f1_macro,
            "F1 Macro",
            "Server Test F1 Macro by Round",
            plots_dir / "f1_macro.png",
        )

    if fit_data:
        fit_rounds = [float(item["round"]) for item in fit_data]
        fit_losses = [float(item["train_loss"]) for item in fit_data]

        _plot_series(
            fit_rounds,
            fit_losses,
            "Train Loss",
            "Aggregated Client Train Loss by Round",
            plots_dir / "train_loss.png",
        )
        _plot_train_loss_boxplot(
            fit_data,
            plots_dir / "train_loss_boxplot.png",
        )

    if train_eval and test_eval:
        _plot_train_test_comparison(
            train_eval,
            test_eval,
            metric_key="loss",
            ylabel="Loss",
            title="Train vs Test Loss by Round",
            out_path=plots_dir / "train_vs_test_loss.png",
        )
        _plot_train_test_comparison(
            train_eval,
            test_eval,
            metric_key="accuracy",
            ylabel="Accuracy",
            title="Train vs Test Accuracy by Round",
            out_path=plots_dir / "train_vs_test_accuracy.png",
        )

    print(f"Plots saved to: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
