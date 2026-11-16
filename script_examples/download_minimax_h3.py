"""Download and verify the pinned models for video_minimax_h3_r2v.json."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


VERSION = "1.0.0"
MANIFEST = Path(__file__).with_name("minimax_h3_models.json")


def verify(path, model):
    if path.stat().st_size != model["size"]:
        raise ValueError(f"Incorrect size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != model["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}; move the file aside before retrying")


def download(root, model, verify_only):
    relative = Path(model["directory"]) / model["name"]
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Model path escapes destination: {relative}")
    if target.exists() or verify_only:
        verify(target, model)
        print(f"VERIFIED {relative}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > model["size"]:
        raise ValueError(f"Partial download exceeds expected size: {partial}")
    if existing < model["size"]:
        print(f"DOWNLOADING {relative}", flush=True)
        command = ["curl.exe", "--fail", "--location", "--silent", "--show-error",
                   "--retry", "5", "--retry-all-errors", "--connect-timeout", "30",
                   "--speed-limit", "1024", "--speed-time", "120",
                   "--continue-at", "-", "--output", str(partial), model["url"]]
        with subprocess.Popen(command) as process:
            while process.poll() is None:
                size = partial.stat().st_size if partial.exists() else 0
                print(f"  {model['name']}: {size / 1e9:.2f}/{model['size'] / 1e9:.2f} GB", flush=True)
                time.sleep(10)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
    print(f"VERIFYING {relative}", flush=True)
    verify(partial, model)
    partial.rename(target)
    print(f"VERIFIED {relative}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("C:/models"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()
    root = args.root.resolve()
    models = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
    if not args.verify_only:
        root.mkdir(parents=True, exist_ok=True)
        remaining = 0
        for model in models:
            target = root / model["directory"] / model["name"]
            partial = target.with_suffix(target.suffix + ".part")
            if not target.exists():
                remaining += max(0, model["size"] - (partial.stat().st_size if partial.exists() else 0))
        if shutil.disk_usage(root).free < remaining + 1024 ** 3:
            raise OSError(f"Need {remaining / 1e9:.2f} GB plus 1 GiB free on {root}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda model: download(root, model, args.verify_only), models))
    print("SUCCESS: all five model files match the pinned publisher SHA-256 checksums.", flush=True)


if __name__ == "__main__":
    main()
