import json
from pathlib import Path

import matplotlib.pyplot as plt


def _plot_series(rounds, values, ylabel, title, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, values, marker="o")
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    history_path = Path("runs/history.json")
    if not history_path.exists():
        raise FileNotFoundError("runs/history.json not found")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    eval_data = history.get("server_evaluate", [])
    fit_data = history.get("fit_metrics", [])
    if not fit_data:
        fit_data = history.get("decentralized_fit", [])

    if not eval_data and not fit_data:
        raise ValueError("No metrics found in history.json")

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if eval_data:
        rounds = [float(item["round"]) for item in eval_data]
        losses = [float(item["loss"]) for item in eval_data]
        accs = [float(item["accuracy"]) for item in eval_data]
        f1_macro = [float(item["f1_macro"]) for item in eval_data]

        _plot_series(
            rounds,
            losses,
            "Loss",
            "Server Evaluation Loss by Round",
            plots_dir / "loss.png",
        )
        _plot_series(
            rounds,
            accs,
            "Accuracy",
            "Server Evaluation Accuracy by Round",
            plots_dir / "accuracy.png",
        )
        _plot_series(
            rounds,
            f1_macro,
            "F1 Macro",
            "Server Evaluation F1 Macro by Round",
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

    print(f"Plots saved to: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
