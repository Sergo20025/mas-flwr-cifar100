import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    history_path = Path("runs/history.json")
    if not history_path.exists():
        raise FileNotFoundError("runs/history.json not found")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    loss_data = history.get("loss_distributed", [])
    acc_data = history.get("metrics_distributed", {}).get("accuracy", [])

    if not loss_data and not acc_data:
        raise ValueError("No metrics found in history.json")

    plots_dir = Path("runs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if loss_data:
        rounds = [item["round"] for item in loss_data]
        losses = [item["loss"] for item in loss_data]

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, losses, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Distributed Loss by Round")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "loss.png")
        plt.close()

    if acc_data:
        rounds = [item["round"] for item in acc_data]
        accs = [item["value"] for item in acc_data]

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, accs, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title("Distributed Accuracy by Round")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "accuracy.png")
        plt.close()

    print(f"Plots saved to: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()