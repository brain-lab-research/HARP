#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import fcntl
from huggingface_hub import HfApi


REQUIRED_LAYER_FILES = ("qkv.pt", "o.pt", "up.pt", "down.pt", "layernorm.pt")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--token-env", default="HF_UPLOAD_TOKEN")
    parser.add_argument("--lock-path", default="/tmp/harp_hf_upload.lock")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(
            f"Missing upload token env var {args.token_env}. "
            f"Example: export {args.token_env}=hf_..."
        )

    folder = Path(args.folder)
    layer = int(args.layer)

    required = [f"{layer}_{suffix}" for suffix in REQUIRED_LAYER_FILES]
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise SystemExit(f"Refusing to upload incomplete layer {layer}; missing: {missing}")

    candidates = list(required)

    timing_name = f"{layer}_timing.json"
    if (folder / timing_name).is_file():
        candidates.append(timing_name)

    # Upload config.pt once if the repo does not already have it.
    if (folder / "config.pt").is_file():
        candidates.append("config.pt")

    lock_path = Path(args.lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        api = HfApi()
        remote_files = set(
            api.list_repo_files(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                token=token,
            )
        )

        # No-overwrite policy: only upload files that do not already exist remotely.
        to_upload = [name for name in candidates if name not in remote_files]

        if not to_upload:
            print(f"Layer {layer}: all artifacts already exist on Hub; skipping upload.")
            return 0

        print(f"Layer {layer}: uploading new artifacts: {to_upload}")

        api.upload_folder(
            folder_path=str(folder),
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            token=token,
            allow_patterns=to_upload,
            commit_message=f"Add HARP quantized artifacts for layer {layer}",
        )

        print(f"Layer {layer}: upload complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())