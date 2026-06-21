#!/usr/bin/env python3
"""
Restore Qdrant database from a backup zip file.

Lists available backups in data/backups/, lets the user choose
(or accepts a filename via CLI). Creates a safety backup of the
current qdrant_db before overwriting, then verifies the restored
database by connecting and checking collection / point count.

Usage:
    python restore_qdrant.py
    python restore_qdrant.py qdrant_backup_20260619_143502.zip
"""

import json
import os
import shutil
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
DEFAULT_QDRANT = (DATA_DIR / "qdrant_db").resolve()


def resolve_qdrant_dir() -> Path:
    """Find the qdrant_db directory from config or default."""
    if BACKEND_CONFIG.exists():
        try:
            with open(BACKEND_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            qdrant_rel = cfg.get("qdrant_path", "..\\data\\qdrant_db")
            qdrant_rel = qdrant_rel.replace("\\", "/")
            resolved = (PROJECT_DIR / "backend" / qdrant_rel).resolve()
            return resolved
        except Exception:
            pass
    return DEFAULT_QDRANT


def get_collection_name() -> str:
    """Read collection name from config or return default."""
    if BACKEND_CONFIG.exists():
        try:
            with open(BACKEND_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("collection", "ebike_manual")
        except Exception:
            pass
    return "ebike_manual"


def list_backups() -> list[Path]:
    """Return sorted list of backup zip files (newest first)."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        BACKUP_DIR.glob("qdrant_backup_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def select_backup(backups: list[Path], cli_arg: str | None = None) -> Path | None:
    """Let user select a backup from the list, or match CLI argument."""
    if not backups:
        print("No backup files found in", BACKUP_DIR)
        return None

    # CLI argument: try to match
    if cli_arg:
        for b in backups:
            if b.name == cli_arg or str(b) == cli_arg:
                return b
        # Try prefix match
        matches = [b for b in backups if b.name.startswith(cli_arg)]
        if len(matches) == 1:
            return matches[0]
        print(f"No backup matches '{cli_arg}'.")
        print(f"Available backups:")
        for i, b in enumerate(backups, 1):
            print(f"  [{i}] {b.name}")
        return None

    # Interactive selection
    print(f"\nAvailable backups ({len(backups)}):")
    print("-" * 60)
    for i, b in enumerate(backups, 1):
        size_mb = b.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(b.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{i}] {b.name}")
        print(f"      {size_mb:.1f} MB  |  {mtime}")
    print("-" * 60)

    while True:
        try:
            choice = input(f"Select backup [1-{len(backups)}] or 'q' to quit: ").strip()
            if choice.lower() == "q":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(backups):
                return backups[idx]
            print(f"Invalid choice. Enter 1-{len(backups)}.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def create_safety_backup(qdrant_dir: Path) -> Path | None:
    """Zip the current qdrant_db as a safety measure. Returns path or None."""
    if not qdrant_dir.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safety_path = BACKUP_DIR / f"qdrant_pre_restore_{timestamp}.zip"
    parent_dir = qdrant_dir.parent

    print(f"\nCreating safety backup of current qdrant_db...")
    file_count = 0
    with zipfile.ZipFile(safety_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(qdrant_dir):
            for fname in files:
                file_path = os.path.join(root, fname)
                arcname = os.path.relpath(file_path, parent_dir)
                zf.write(file_path, arcname)
                file_count += 1

    size_mb = safety_path.stat().st_size / (1024 * 1024)
    print(f"Safety backup created: {safety_path.name} ({file_count} files, {size_mb:.1f} MB)")
    return safety_path


def restore_from_zip(zip_path: Path, target_parent: Path, target_name: str = "qdrant_db") -> bool:
    """Extract zip to target_parent, expecting entries under target_name/.
    Removes existing target_name directory first."""
    target_dir = target_parent / target_name

    # Remove existing
    if target_dir.exists():
        print(f"Removing existing {target_dir} ...")
        shutil.rmtree(target_dir)

    # Extract
    print(f"Extracting {zip_path.name} ...")
    file_count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            zf.extract(member, target_parent)
            file_count += 1
            if file_count % 200 == 0:
                print(f"  {file_count} files...")

    print(f"Extracted {file_count} files to {target_dir}")
    return target_dir.exists()


def verify_qdrant(qdrant_dir: Path, collection_name: str) -> dict | None:
    """Connect to Qdrant and verify collection. Returns info dict or None on failure."""
    print(f"\nVerifying Qdrant at {qdrant_dir} ...")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(qdrant_dir))
        info = client.get_collection(collection_name)
        points_count = info.points_count
        client.close()

        result = {
            "collection": collection_name,
            "points": points_count,
            "status": "ok",
        }
        print(f"  Collection: {collection_name}")
        print(f"  Points:     {points_count}")
        print(f"  Status:     ok")
        return result
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    print("=" * 60)
    print("Qdrant Restore Tool")
    print("=" * 60)

    qdrant_dir = resolve_qdrant_dir()
    collection_name = get_collection_name()

    print(f"Target Qdrant dir: {qdrant_dir}")
    print(f"Collection:        {collection_name}")

    # Show current state
    if qdrant_dir.exists():
        file_count = sum(1 for _ in qdrant_dir.rglob("*") if _.is_file())
        print(f"Current state:     {file_count} files on disk")
    else:
        print(f"Current state:     directory does not exist")

    # List and select backup
    backups = list_backups()
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    selected = select_backup(backups, cli_arg)

    if selected is None:
        print("No backup selected. Exiting.")
        sys.exit(0)

    print(f"\nSelected: {selected.name}")
    print(f"Size:     {selected.stat().st_size / (1024 * 1024):.1f} MB")

    # Confirm
    try:
        confirm = input(f"\nRestore from this backup? This will overwrite the current qdrant_db. [y/N]: ").strip().lower()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)

    if confirm != "y":
        print("Cancelled.")
        sys.exit(0)

    # Step 1: Safety backup of current qdrant_db
    safety_path = create_safety_backup(qdrant_dir)

    # Step 2: Restore from selected backup
    success = restore_from_zip(selected, qdrant_dir.parent, qdrant_dir.name)

    if not success:
        print("\nERROR: Restore failed — target directory not created after extraction.")
        if safety_path:
            print(f"Your data was backed up to: {safety_path}")
            print(f"To recover: unzip {safety_path.name} to {qdrant_dir.parent}")
        sys.exit(1)

    # Step 3: Verify
    result = verify_qdrant(qdrant_dir, collection_name)

    # Step 4: Report
    print("\n" + "=" * 60)
    if result and result["points"] > 0:
        print("SUCCESS: Qdrant restored and verified.")
        print(f"  Collection: {result['collection']}")
        print(f"  Points:     {result['points']}")
        # Remove safety backup since restore succeeded
        if safety_path and safety_path.exists():
            safety_path.unlink()
            print(f"  Safety backup removed (restore successful)")
    elif result and result["points"] == 0:
        print("WARNING: Qdrant restored but collection has 0 points.")
        print(f"  Safety backup preserved: {safety_path}")
    else:
        print("WARNING: Qdrant verification failed.")
        print(f"  Safety backup preserved: {safety_path}")
        print(f"  To manually recover: delete {qdrant_dir}")
        print(f"    then unzip {safety_path.name} in {qdrant_dir.parent}")
    print("=" * 60)


if __name__ == "__main__":
    main()
