"""Resolve the selected veRL source without requiring pip distribution metadata."""

import importlib
import os
import sys
from pathlib import Path


REQUIRED_VERL_VERSION = "0.9.0.dev"


def validate_verl_version(selected_version, expected_version=REQUIRED_VERL_VERSION):
    """Accept the server dev version and its equivalent dev0/local-build spelling."""
    def public_version(value):
        public = str(value).strip().split("+", 1)[0]
        return public + "0" if public.endswith(".dev") else public

    if public_version(selected_version) != public_version(expected_version):
        raise RuntimeError(
            f"Shopping integration requires veRL {expected_version} source; found {selected_version!r}"
        )


def resolve_verl_source(expected_version=REQUIRED_VERL_VERSION):
    """Return (version, __init__.py) from the package this Python actually imports."""
    try:
        verl = importlib.import_module("verl")
    except ModuleNotFoundError as exc:
        if exc.name == "verl":
            raise RuntimeError(
                f"cannot import verl in {sys.executable}; set VERL_ROOT to the checkout "
                "containing verl/__init__.py and add that checkout to PYTHONPATH"
            ) from exc
        raise RuntimeError(f"cannot import verl because a dependency is missing: {exc}") from exc
    except ImportError as exc:
        raise RuntimeError(f"cannot import the selected verl source: {exc}") from exc

    if not getattr(verl, "__file__", None):
        raise RuntimeError("imported verl has no __file__; select a complete veRL source checkout")
    source = Path(verl.__file__).resolve()
    if os.environ.get("VERL_ROOT"):
        expected_source = Path(os.environ["VERL_ROOT"]).expanduser().resolve() / "verl/__init__.py"
        if source != expected_source:
            raise RuntimeError(f"VERL_ROOT selection mismatch: imported {source}, expected {expected_source}")

    selected_version = getattr(verl, "__version__", None)
    if not isinstance(selected_version, str) or not selected_version.strip():
        version_file = source.parent / "version/version"
        try:
            selected_version = version_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                f"cannot determine veRL source version: no __version__ or readable {version_file}"
            ) from exc
    selected_version = selected_version.strip()
    try:
        validate_verl_version(selected_version, expected_version)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} from {source}") from exc
    return selected_version, source
