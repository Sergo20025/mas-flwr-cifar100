from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "uoft-cs/cifar100"


def save_split(dataset_split, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(dataset_split):
        img = item["img"]
        label = item["fine_label"]

        class_dir = output_dir / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)

        idx = item["img"].filename if hasattr(item["img"], "filename") else None

        filename = class_dir / f"{hash(img.tobytes())}.png"

        img.save(filename)


def main():

    root = Path("data")
    dataset_dir = root / "cifar100"

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    print("Downloading dataset from HuggingFace...")

    dataset = load_dataset(DATASET_NAME)

    print("Saving train split...")
    save_split(dataset["train"], train_dir)

    print("Saving test split...")
    save_split(dataset["test"], test_dir)

    print("Dataset saved to:", dataset_dir)


if __name__ == "__main__":
    main()