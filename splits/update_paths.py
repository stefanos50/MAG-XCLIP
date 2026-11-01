import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("txt_dir", type=Path)
parser.add_argument("normal_path", type=Path)
parser.add_argument("violence_path", type=Path)
args = parser.parse_args()

paths = {"0": args.normal_path, "1": args.violence_path}

for txt_file in args.txt_dir.glob("*.txt"):
    updated = []

    for line in txt_file.read_text(encoding="utf-8").splitlines(keepends=True):
        parts = line.rsplit(maxsplit=1)

        if len(parts) != 2 or parts[1] not in paths:
            updated.append(line)
            continue

        video_path, label = parts
        new_path = paths[label] / Path(video_path).name
        ending = "\n" if line.endswith("\n") else ""
        updated.append(f"{new_path} {label}{ending}")

    txt_file.write_text("".join(updated), encoding="utf-8")
    print(f"Updated: {txt_file}")