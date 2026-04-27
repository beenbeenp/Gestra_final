"""Download a subset of HMDB51 from the HuggingFace mirror.

We only pull the four classes we need (punch, kick, kick_ball, stand) using
HTTP range requests via `remotezip`, so we don't have to download the full
2.1 GB archive. Output layout:

    data/videos/punch/<clip>.avi
    data/videos/kick/<clip>.avi
    data/videos/kick_ball/<clip>.avi
    data/videos/stand/<clip>.avi

Re-running is idempotent: existing files are skipped.
"""

import argparse
import os
import sys
from pathlib import Path

from remotezip import RemoteZip
from tqdm import tqdm


HMDB51_URL = "https://huggingface.co/datasets/jili5044/hmdb51/resolve/main/hmdb51.zip"
DEFAULT_CLASSES = ("punch", "kick", "kick_ball", "stand")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/videos"),
                   help="Output directory for class subfolders.")
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES),
                   help="HMDB51 class names to download.")
    p.add_argument("--url", default=HMDB51_URL,
                   help="Source zip URL (HuggingFace mirror by default).")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="Optional cap on clips per class (for quick smoke runs).")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Source: {args.url}")
    print(f"Classes: {args.classes}")
    print(f"Output : {args.out.resolve()}")

    with RemoteZip(args.url) as zf:
        all_names = zf.namelist()
        for cls in args.classes:
            cls_dir = args.out / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            members = [n for n in all_names
                       if n.startswith(f"hmdb51/{cls}/") and not n.endswith("/")]
            if args.max_per_class is not None:
                members = members[: args.max_per_class]
            print(f"\n[{cls}] {len(members)} clips queued")

            for name in tqdm(members, desc=cls, unit="clip"):
                # Strip the "hmdb51/<cls>/" prefix and write to cls_dir.
                base = os.path.basename(name)
                target = cls_dir / base
                if target.exists() and target.stat().st_size > 0:
                    continue
                try:
                    with zf.open(name) as src, open(target, "wb") as dst:
                        while True:
                            chunk = src.read(1 << 20)
                            if not chunk:
                                break
                            dst.write(chunk)
                except Exception as exc:
                    print(f"  WARN: failed {name}: {exc}", file=sys.stderr)
                    if target.exists():
                        target.unlink()

    print("\nDone.")
    for cls in args.classes:
        cls_dir = args.out / cls
        n = sum(1 for _ in cls_dir.glob("*"))
        print(f"  {cls}: {n} clips on disk")


if __name__ == "__main__":
    main()
