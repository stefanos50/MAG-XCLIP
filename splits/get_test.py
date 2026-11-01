import argparse
import os
import shutil


def load_split(file_path):
    video_paths, labels = [], []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            path, label = line.rsplit(maxsplit=1)
            video_paths.append(path)
            labels.append(int(label))

    return video_paths, labels


def organize_videos(file_path, output_root):
    video_paths, labels = load_split(file_path)

    normal_dir = os.path.join(output_root, "normal")
    violence_dir = os.path.join(output_root, "violence")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(violence_dir, exist_ok=True)

    for path, label in zip(video_paths, labels):
        if label not in (0, 1):
            print(f"Warning: Invalid label {label} -> {path}")
            continue

        if not os.path.isfile(path):
            print(f"Warning: File not found -> {path}")
            continue

        dest_dir = normal_dir if label == 0 else violence_dir
        dest_path = os.path.join(dest_dir, os.path.basename(path))

        shutil.copy2(path, dest_path)
        print(f"Copied {path} -> {dest_path}")

    print("Finished organizing videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy videos into normal and violence directories."
    )
    parser.add_argument("--txt-file", required=True, help="Input TXT file")
    parser.add_argument(
        "--output-root",
        default="./DATASET/validation/",
        help="Output directory (default: ./DATASET/validation/)",
    )
    args = parser.parse_args()

    organize_videos(args.txt_file, args.output_root)