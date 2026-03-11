import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split video pretrain file list.")
    parser.add_argument(
        "--input",
        default=None,
        help="Input txt file path. Defaults to DataPreProcess/video_pretrain.txt",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to DataPreProcess/video_pretrain",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input) if args.input else script_dir / "video_pretrain.txt"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "video_pretrain"

    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(args.seed)
    random.shuffle(lines)

    num_train = int(len(lines) * args.train_ratio)
    train_lines = lines[:num_train]
    val_lines = lines[num_train:]

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"划分完成：训练集 {len(train_lines)} 条，验证集 {len(val_lines)} 条")
    print(f"train.txt: {train_path}")
    print(f"val.txt: {val_path}")


if __name__ == "__main__":
    main()
