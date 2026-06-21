#!/usr/bin/env python3
"""
Backup Qdrant database directory to a timestamped zip file.

Reads config.json to locate qdrant_db, compresses it into
data/backups/qdrant_backup_YYYYMMDD_HHMMSS.zip, and keeps only
the 5 most recent backups.

Usage: python backup_qdrant.py
"""

import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BACKEND_CONFIG = PROJECT_DIR / "backend" / "config.json"
DATA_DIR = PROJECT_DIR / "data"
BACKUP_DIR = DATA_DIR / "backups"


def resolve_qdrant_dir() -> Path:
    """Find the qdrant_db directory from config or default."""
    if BACKEND_CONFIG.exists():
        try:
            with open(BACKEND_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            qdrant_rel = cfg.get("qdrant_path", "..\\data\\qdrant_db")
            # Normalize Windows backslashes to forward for pathlib
            qdrant_rel = qdrant_rel.replace("\\", "/")
            resolved = (PROJECT_DIR / "backend" / qdrant_rel).resolve()
            if resolved.exists():
                return resolved
        except Exception as e:
            print(f"WARNING: Failed to read config: {e}")

    # Fallback: default relative to project root
    fallback = (DATA_DIR / "qdrant_db").resolve()
    if fallback.exists():
        return fallback

    # Last resort
    return fallback


def backup_qdrant(qdrant_dir: Path, backup_dir: Path) -> Path:
    """Compress qdrant_dir into a timestamped zip, return the zip path."""
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"qdrant_backup_{timestamp}.zip"
    zip_path = backup_dir / zip_name

    parent_dir = qdrant_dir.parent  # archive entries relative to parent

    file_count = 0
    print(f"Compressing {qdrant_dir} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(qdrant_dir):
            for fname in files:
                file_path = os.path.join(root, fname)
                arcname = os.path.relpath(file_path, parent_dir)
                zf.write(file_path, arcname)
                file_count += 1
                if file_count % 50 == 0:
                    print(f"  {file_count} files...")

    return zip_path, file_count


def cleanup_old_backups(backup_dir: Path, keep: int = 5):
    """Remove old backups, keeping only the most recent `keep`."""
    pattern = "qdrant_backup_*.zip"
    backups = sorted(
        backup_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in backups[keep:]:
        print(f"Removing old backup: {old.name}")
        old.unlink()


def main():
    print("=" * 60)
    print("Qdrant Backup Tool")
    print("=" * 60)

    # Locate qdrant_db
    qdrant_dir = resolve_qdrant_dir()
    if not qdrant_dir.exists():
        print(f"ERROR: qdrant_db not found at {qdrant_dir}")
        sys.exit(1)
    print(f"Source: {qdrant_dir}")

    # Calculate directory size before backup
    total_size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(qdrant_dir)
        for f in files
    )
    print(f"Size on disk: {total_size / (1024 * 1024):.1f} MB")

    # Create backup
    zip_path, file_count = backup_qdrant(qdrant_dir, BACKUP_DIR)

    # Report
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nBackup complete!")
    print(f"  File: {zip_path}")
    print(f"  Files: {file_count}")
    print(f"  Compressed size: {zip_size_mb:.1f} MB")
    if total_size > 0:
        ratio = (1 - zip_path.stat().st_size / total_size) * 100
        print(f"  Compression: {ratio:.0f}% space saved")

    # Cleanup old backups
    cleanup_old_backups(BACKUP_DIR, keep=5)

    # List remaining backups
    remaining = sorted(BACKUP_DIR.glob("qdrant_backup_*.zip"))
    if remaining:
        print(f"\nBackups ({len(remaining)} total):")
        for b in remaining:
            size_mb = b.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(b.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {b.name}  ({size_mb:.1f} MB, {mtime})")


if __name__ == "__main__":
    main()
