"""Dataset source registry and downloader for SignVision (design sections 4.1, 4.2).

REGISTRY lists every dataset the YOLO26 pipeline can use. ``fetch`` downloads
one source and never raises:
  - kaggle sources go through kagglehub (cached under ~/.cache/kagglehub). Any
    failure, most often missing Kaggle credentials, becomes "skipped" with a
    hint naming the credential to add;
  - url_zip sources stream to data/raw/<name>.zip.part, are validated (size,
    zip structure, CRCs), then extracted to data/raw/<name>/ with a .complete
    marker. Any failure deletes the partial files and becomes "failed".

Every ok source, fresh or cached, is merged into data/sources.json
({name: {"path", "fetched_at"}}), which the dataset builders read.

The exit code is 0 iff every requested *required* source is ok. Optional
sources never cause a non-zero exit, and no traceback reaches the user.

Run:
    .venv/bin/python -m src.download_data [--sources a,b,...] [--list]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Allow running directly (python src/download_data.py) by putting the project
# root on sys.path so the `src` package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402

CHUNK_BYTES = 1 << 20  # 1 MiB streaming chunks for url_zip downloads
URL_TIMEOUT_S = 60

CREDENTIAL_HINT = (
    "Kaggle download failed; add a Kaggle API token and re-run: kaggle.com -> Settings -> API -> "
    "Create New Token, save it to ~/.kaggle/kaggle.json (chmod 600) or set KAGGLE_USERNAME + KAGGLE_KEY"
)


@dataclass(frozen=True)
class Source:
    name: str
    kind: str  # "kaggle" | "url_zip"
    ref: str  # Kaggle handle or zip URL
    role: str  # "cls_train" | "cls_test" | "det"
    required: bool
    license: str
    size_mb: int
    class_root: str | None = None  # class-folder root relative to the download root
    test_glob: str | None = None  # loose labeled test files relative to the download root


REGISTRY = OrderedDict(
    (s.name, s)
    for s in (
        Source(
            "grassknoted", "kaggle", "grassknoted/asl-alphabet", "cls_train", True, "GPL-2", 1100,
            class_root="asl_alphabet_train/asl_alphabet_train",
            test_glob="asl_alphabet_test/asl_alphabet_test/*_test.jpg",
        ),
        Source(
            "debashishsau", "kaggle", "debashishsau/aslamerican-sign-language-aplhabet-dataset",
            "cls_train", False, "CC0", 4600,
            class_root="ASL_Alphabet_Dataset/asl_alphabet_train",
        ),
        Source("ayuraj", "kaggle", "ayuraj/asl-dataset", "cls_train", False, "CC0", 32),
        Source("danrasband", "kaggle", "danrasband/asl-alphabet-test", "cls_test", False, "CC0", 13),
        Source(
            "hand_keypoints", "url_zip",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip",
            "det", True, "CC BY-NC-SA 4.0", 387,
        ),
        Source(
            "hagrid30k", "kaggle", "innominate817/hagrid-sample-30k-384p", "det", False, "CC BY-SA 4.0", 1000,
        ),
    )
)


@dataclass(frozen=True)
class FetchResult:
    name: str
    status: str  # "ok" | "skipped" | "failed"
    path: Path | None
    message: str


class _DownloadError(Exception):
    """A url_zip download that arrived but failed validation."""


def fetch(source, raw_dir=None):
    """Download one source; never raises. Returns a FetchResult."""
    if source.kind == "kaggle":
        return _fetch_kaggle(source)
    if source.kind == "url_zip":
        return _fetch_url_zip(source, Path(raw_dir) if raw_dir is not None else config.RAW_DIR)
    return FetchResult(source.name, "failed", None, f"unknown source kind {source.kind!r}")


def _fetch_kaggle(source):
    try:
        import kagglehub

        path = kagglehub.dataset_download(source.ref)
    except Exception as exc:  # noqa: BLE001 - a skipped source is never fatal and never a traceback
        return FetchResult(source.name, "skipped", None, f"{CREDENTIAL_HINT} ({type(exc).__name__}: {exc})")
    return FetchResult(source.name, "ok", Path(path), "downloaded or cached by kagglehub")


def _fetch_url_zip(source, raw_dir):
    dest = raw_dir / source.name
    marker = dest / ".complete"
    if marker.exists():
        return FetchResult(source.name, "ok", dest, "already present")

    part = raw_dir / f"{source.name}.zip.part"
    zip_path = raw_dir / f"{source.name}.zip"
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(source.ref, timeout=URL_TIMEOUT_S) as resp:
            length = resp.headers.get("Content-Length")
            expected = int(length) if length else None
            got = _stream_to_file(resp, part, expected, source.name)
        if expected is not None and got != expected:
            raise _DownloadError(f"size mismatch: received {got} bytes, Content-Length was {expected}")
        if not zipfile.is_zipfile(part):
            raise _DownloadError("downloaded file is not a zip archive")
        with zipfile.ZipFile(part) as zf:
            bad = zf.testzip()
        if bad is not None:
            raise _DownloadError(f"corrupt zip member {bad!r}")
        os.replace(part, zip_path)
        if dest.exists():
            shutil.rmtree(dest)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        marker.write_text(datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n")
        zip_path.unlink()
    except Exception as exc:  # noqa: BLE001 - network, size or zip errors all become "failed"
        part.unlink(missing_ok=True)
        zip_path.unlink(missing_ok=True)
        shutil.rmtree(dest, ignore_errors=True)
        return FetchResult(source.name, "failed", None, f"{type(exc).__name__}: {exc}")
    return FetchResult(source.name, "ok", dest, f"downloaded {got / 1e6:.0f} MB and extracted")


def _stream_to_file(resp, path, expected, name):
    """Copy the response body to ``path`` in 1 MiB chunks; return the byte count."""
    got = 0
    next_report = 0.1
    with open(path, "wb") as fh:
        while True:
            chunk = resp.read(CHUNK_BYTES)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if expected and got / expected >= next_report:
                print(f"[{name}] {got / 1e6:.0f}/{expected / 1e6:.0f} MB", file=sys.stderr, flush=True)
                next_report += 0.1
    return got


def merge_sources_json(results, path=None):
    """Merge every ok result into sources.json atomically; return the merged dict."""
    path = Path(path) if path is not None else config.SOURCES_FILE
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: {path} is unreadable ({exc}); rewriting it", file=sys.stderr)
            data = {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for result in results:
        if result.status == "ok":
            data[result.name] = {"path": str(result.path), "fetched_at": now}

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return data


def read_sources(path=None):
    """Return {name: Path} for every source recorded in sources.json ({} if absent)."""
    path = Path(path) if path is not None else config.SOURCES_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {name: Path(entry["path"]) for name, entry in data.items()}


def sources_arg(role=None):
    """argparse type for --sources: comma-separated registry names (optionally role-filtered)."""
    allowed = [name for name, src in REGISTRY.items() if role is None or src.role == role]

    def parse(text):
        names = [part.strip() for part in text.split(",") if part.strip()]
        if not names:
            raise argparse.ArgumentTypeError("expected a comma-separated list of source names")
        for name in names:
            if name not in REGISTRY:
                raise argparse.ArgumentTypeError(f"unknown source {name!r}; choose from {', '.join(allowed)}")
            if name not in allowed:
                raise argparse.ArgumentTypeError(
                    f"source {name!r} has role {REGISTRY[name].role!r}; this command accepts only "
                    f"{role!r} sources: {', '.join(allowed)}"
                )
        return list(dict.fromkeys(names))

    return parse


def is_cached(source, raw_dir=None):
    """Cheap on-disk check (no network) used by --list."""
    if source.kind == "url_zip":
        raw_dir = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
        return (raw_dir / source.name / ".complete").exists()
    cache = Path(os.environ.get("KAGGLEHUB_CACHE", Path.home() / ".cache" / "kagglehub"))
    owner, slug = source.ref.split("/", 1)
    return any((cache / "datasets" / owner / slug).glob("*.complete"))


def print_registry():
    header = f"{'name':<15} {'role':<10} {'size_mb':>7}  {'license':<16} {'required':<8}  cached"
    print(header)
    for src in REGISTRY.values():
        print(
            f"{src.name:<15} {src.role:<10} {src.size_mb:>7}  {src.license:<16} "
            f"{'yes' if src.required else 'no':<8}  {'yes' if is_cached(src) else 'no'}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.download_data",
        description="Download the SignVision dataset sources and record them in data/sources.json.",
    )
    parser.add_argument(
        "--sources",
        type=sources_arg(),
        default=list(REGISTRY),
        help=f"comma-separated source names (default: all of {','.join(REGISTRY)})",
    )
    parser.add_argument("--list", action="store_true", help="print the source registry and exit")
    args = parser.parse_args(argv)

    if args.list:
        print_registry()
        return 0

    results = []
    for name in args.sources:
        src = REGISTRY[name]
        print(f"[{name}] fetching {src.kind} {src.ref} (~{src.size_mb} MB) ...", file=sys.stderr, flush=True)
        results.append(fetch(src))
    merge_sources_json(results)

    print("download summary:", file=sys.stderr)
    for result in results:
        src = REGISTRY[result.name]
        where = f" -> {result.path}" if result.path else ""
        need = "required" if src.required else "optional"
        print(f"  {result.name:<15} {result.status:<7} ({need}) {result.message}{where}", file=sys.stderr)

    missing = [r.name for r in results if REGISTRY[r.name].required and r.status != "ok"]
    if missing:
        print(
            f"ERROR: required source(s) not available: {', '.join(missing)}; fix the cause above and re-run "
            f".venv/bin/python -m src.download_data --sources {','.join(missing)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
