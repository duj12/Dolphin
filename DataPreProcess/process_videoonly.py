import argparse
import os


DEFAULT_IN_DIRS = [
    "lrs2/mouths",
    "lrs3_rebuild/mouths",
    "vox2/mouths",
]


def preprocess(args):
    """Collect all file paths from input dirs into one text file."""
    total = 0
    with open(args.out_dir, "w", encoding="utf-8") as f:
        for in_dir in args.in_dirs:
            if not os.path.isdir(in_dir):
                print(f"[WARN] Skip non-existing dir: {in_dir}")
                continue

            for filename in sorted(os.listdir(in_dir)):
                f.write(os.path.join(in_dir, filename))
                f.write("\n")
                total += 1
    print(f"[INFO] Wrote {total} entries to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Video data preprocessing")
    parser.add_argument(
        "--in_dirs",
        nargs="+",
        default=DEFAULT_IN_DIRS,
        help="Input video directories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="video_pretrain.txt",
        help="Output txt file path.",
    )
    args = parser.parse_args()
    print(args)
    preprocess(args)
