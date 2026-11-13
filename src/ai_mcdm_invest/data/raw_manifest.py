"""The raw-data manifest: size, SHA-256 and date of every raw file the build reads.

Raw sources change after the fact: companies restate figures, the SEC labels
old filings with today's company names, and the ticker list changes daily. So
the study is tied to one recorded copy of the raw files. `make manifest`
records it in `raw_manifest.json` (committed), and every rebuild first checks
the raw folder against it and stops on any difference.

Run `uv run python -m ai_mcdm_invest.data.raw_manifest` (or `make manifest`)
only to adopt a new copy of the raw data on purpose; the change then shows up
in git.
"""

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

from ai_mcdm_invest.config import Settings


class RawDataChanged(Exception):
    """The raw folder no longer holds the files the manifest recorded."""


def raw_files_read(settings: Settings) -> list[str]:
    """Every file in the raw folders the build reads, relative to `settings.raw_dir`."""
    folders = [
        settings.raw_panel_dir,
        settings.raw_sec_dir,
        settings.raw_frames_dir,
        settings.raw_corpus_dir,
    ]
    paths = [path for folder in folders for path in folder.iterdir() if path.is_file()]
    return sorted(path.relative_to(settings.raw_dir).as_posix() for path in paths)


def record_manifest(raw_dir: Path, files: list[str]) -> dict:
    """The manifest of `files` (paths relative to `raw_dir`) as they are now."""
    entries = {}
    for name in sorted(files):
        path = raw_dir / name
        modified = datetime.fromtimestamp(path.stat().st_mtime, UTC).date()
        entries[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "modified": modified.isoformat(),
        }
    return {"recorded": date.today().isoformat(), "files": entries}


def verify_raw_files(raw_dir: Path, manifest: dict) -> None:
    """Raise `RawDataChanged` unless `raw_dir` holds exactly the recorded files.

    Every listed file must exist with the recorded size and hash, and a folder
    holding listed files must hold no others: stages read whole folders, so an
    extra file would enter the study unverified.
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"raw data folder not found: {raw_dir}. "
            "Set AI_MCDM_RAW_DIR to the folder that contains it."
        )
    listed = manifest["files"]
    problems = []
    for name, entry in listed.items():
        path = raw_dir / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            problems.append(f"changed: {name}")
    folders = {(raw_dir / name).parent for name in listed}
    for folder in sorted(f for f in folders if f.is_dir()):
        for path in sorted(p for p in folder.iterdir() if p.is_file()):
            name = path.relative_to(raw_dir).as_posix()
            if name not in listed:
                problems.append(f"not in the manifest: {name}")
    if problems:
        raise RawDataChanged(
            f"raw data in {raw_dir} differs from the manifest:\n  "
            + "\n  ".join(problems)
            + "\nRestore the recorded files, or run `make manifest` to adopt new ones on purpose."
        )


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"raw-data manifest not found: {path}. Run `make manifest`.")
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    settings = Settings.from_env()
    manifest = record_manifest(settings.raw_dir, raw_files_read(settings))
    settings.manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"recorded {len(manifest['files'])} raw files in {settings.manifest_path}")


if __name__ == "__main__":
    main()
