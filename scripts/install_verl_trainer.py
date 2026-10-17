#!/usr/bin/env python3
"""Install the reviewed server trainer directly into the active veRL source.

This replaces one known 0.9.0.dev source file; it does not install veRL, invoke
the old 0.8 patch, or modify other files in the working server environment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_FILE = PROJECT_ROOT / "integrations/verl/ray_trainer.py"
BACKUP_SUFFIX = ".shopping-grpo-trainer.orig"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shopping_grpo.training.grpo.trainer_contract import (
    EXPECTED_ORIGINAL_SHA256,
    EXPECTED_TRAINER_SHA256,
    normalized_sha256,
    normalized_source_bytes,
)
from shopping_grpo.training.grpo.verl_source import resolve_verl_source


def resolve_installed_ray_trainer() -> Path:
    selected_version, verl_source = resolve_verl_source()
    print(f"veRL source: {verl_source}, version={selected_version} (pip metadata not required)")
    target = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    if not target.is_file():
        raise RuntimeError(f"installed ray_trainer.py does not exist: {target}")
    return target.resolve()


def _verify_source(path: Path, expected_hash: str, description: str) -> bytes:
    actual_hash = normalized_sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{description} hash mismatch: {path}; expected {expected_hash}, got {actual_hash}"
        )
    source = normalized_source_bytes(path)
    compile(source, str(path), "exec")
    return source


def verify_installed(target: Path) -> None:
    _verify_source(target, EXPECTED_TRAINER_SHA256, "Shopping trainer")


def _replace_source(target: Path, source: bytes) -> None:
    """Publish a complete file on the same filesystem, preserving permissions."""
    compile(source, str(target), "exec")
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.shopping-", dir=target.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copystat(target, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _backup_original(target: Path, backup: Path) -> None:
    if backup.exists():
        _verify_source(backup, EXPECTED_ORIGINAL_SHA256, "original backup")
        return
    # Exclusive creation preserves an existing backup, including concurrent installs.
    with backup.open("xb") as handle:
        try:
            handle.write(target.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            handle.close()
            backup.unlink(missing_ok=True)
            raise
    shutil.copystat(target, backup)
    _verify_source(backup, EXPECTED_ORIGINAL_SHA256, "original backup")


def install_trainer(target: Path) -> None:
    source = _verify_source(TRAINER_FILE, EXPECTED_TRAINER_SHA256, "repository trainer")
    target_hash = normalized_sha256(target)
    backup = Path(str(target) + BACKUP_SUFFIX)
    if backup.exists():
        _verify_source(backup, EXPECTED_ORIGINAL_SHA256, "original backup")
    if target_hash == EXPECTED_TRAINER_SHA256:
        verify_installed(target)
        print(f"Shopping trainer already installed: {target}")
        return
    if target_hash != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            f"refusing to overwrite unknown ray_trainer.py: {target}; "
            f"expected original SHA256 {EXPECTED_ORIGINAL_SHA256}, got {target_hash}. "
            "Provide this source for review; the installer only accepts the reviewed server trainer."
        )
    _backup_original(target, backup)
    # Catch a file changed after the initial read, before replacing server source.
    _verify_source(target, EXPECTED_ORIGINAL_SHA256, "current original trainer")
    try:
        _replace_source(target, source)
        verify_installed(target)
    except Exception:
        if normalized_sha256(target) != EXPECTED_ORIGINAL_SHA256:
            _replace_source(target, backup.read_bytes())
        raise
    print(f"installed Shopping trainer: {target}")
    print(f"original backup: {backup}")
    print(f"trainer_sha256: {normalized_sha256(target)}")


def restore_trainer(target: Path) -> None:
    target_hash = normalized_sha256(target)
    if target_hash == EXPECTED_ORIGINAL_SHA256:
        _verify_source(target, EXPECTED_ORIGINAL_SHA256, "original trainer")
        print(f"veRL ray_trainer.py is already original: {target}")
        return
    if target_hash != EXPECTED_TRAINER_SHA256:
        raise RuntimeError(f"refusing to restore over unknown ray_trainer.py: {target}")
    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.is_file():
        raise RuntimeError(f"cannot restore without original backup: {backup}")
    _verify_source(backup, EXPECTED_ORIGINAL_SHA256, "original backup")
    _replace_source(target, backup.read_bytes())
    _verify_source(target, EXPECTED_ORIGINAL_SHA256, "restored original trainer")
    print(f"restored original veRL ray_trainer.py: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify installed source without writing files")
    mode.add_argument("--restore", action="store_true", help="restore the verified original backup")
    args = parser.parse_args()
    try:
        target = resolve_installed_ray_trainer()
        if args.check:
            verify_installed(target)
            print(f"verified Shopping trainer: {target}")
        elif args.restore:
            restore_trainer(target)
        else:
            install_trainer(target)
    except (OSError, RuntimeError, SyntaxError) as exc:
        raise SystemExit(f"Shopping trainer installation error: {exc}") from exc


if __name__ == "__main__":
    main()
