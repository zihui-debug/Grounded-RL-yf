#!/usr/bin/env python3
"""
download_and_extract.py

Downloads your VIGORL dataset tarballs from Hugging Face (tracking
dataset downloads in HF metrics), extracts them under DATA_ROOT, and
removes the tarballs.

Usage:
    export DATA_ROOT=/path/to/data
    python download_and_extract.py
"""

import os
import sys
import tarfile
from huggingface_hub import snapshot_download

# -----------------------------------------------------------------------------
# 1️⃣ Check DATA_ROOT
# -----------------------------------------------------------------------------
DATA_ROOT = os.getenv("DATA_ROOT")
if not DATA_ROOT:
    sys.stderr.write("Error: DATA_ROOT environment variable is not set.\n")
    sys.stderr.write("Please set DATA_ROOT to the directory where you want to store the data.\n")
    sys.exit(1)

os.makedirs(DATA_ROOT, exist_ok=True)
print(f"✅ DATA_ROOT is set to: {DATA_ROOT}")

# -----------------------------------------------------------------------------
# 2️⃣ Download the entire dataset snapshot (counts as a HF dataset download)
# -----------------------------------------------------------------------------
print("\n🔄 Downloading dataset snapshot...")
snapshot_download(
    repo_id="gsarch/vigorl_datasets",
    repo_type="dataset",
    local_dir=DATA_ROOT,
    local_dir_use_symlinks=False,   # ensures real files, not symlinks
)

# -----------------------------------------------------------------------------
# 3️⃣ Extract each tarball and clean up
# -----------------------------------------------------------------------------
datasets = [
    "spatial_reasoning",
    "visual_search",
    "web_action",
    "web_grounding",
]

for ds in datasets:
    tar_path = os.path.join(DATA_ROOT, f"{ds}.tar")
    if not os.path.isfile(tar_path):
        print(f"⚠️  Warning: {tar_path} not found, skipping.")
        continue

    print(f"\n📂 Extracting {ds}.tar …")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=DATA_ROOT)

    print(f"🧹 Removing {ds}.tar …")
    os.remove(tar_path)

print("\n🎉 All done! Your data folders are ready under:")
for ds in datasets:
    print(f" • {os.path.join(DATA_ROOT, ds)}")
