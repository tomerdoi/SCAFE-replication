"""Download and verify bounded official Solidity compiler binaries.

The official build manifest is fetched once over HTTPS.  Every binary is
executed only after its SHA-256 matches the corresponding manifest entry.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess


BASE_URL = "https://binaries.soliditylang.org/macosx-amd64/"
VERSION_RE = re.compile(r"0\.[0-9]+\.[0-9]+")
MAX_LIST_BYTES = 2_000_000
MAX_BINARY_BYTES = 80_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, destination: Path, size_limit: int) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    part = destination.with_suffix(destination.suffix + ".part")
    if part.exists() or part.is_symlink():
        raise FileExistsError(part)
    process = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--max-redirs",
            "0",
            "--max-time",
            "180",
            "--max-filesize",
            str(size_limit),
            "--output",
            str(part),
            url,
        ],
        capture_output=True,
        text=True,
        timeout=190,
        check=False,
    )
    if process.returncode:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {process.stderr.strip()}")
    if not part.is_file() or part.stat().st_size > size_limit:
        part.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file violates size bound: {url}")
    receipt = {
        "url": url,
        "bytes": part.stat().st_size,
        "sha256": _sha256(part),
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    part.rename(destination)
    return receipt


def _select_build(listing: dict[str, object], version: str) -> dict[str, object]:
    releases = listing.get("releases")
    builds = listing.get("builds")
    if not isinstance(releases, dict) or not isinstance(builds, list):
        raise ValueError("Malformed official compiler manifest")
    path = releases.get(version)
    matches = [
        entry
        for entry in builds
        if isinstance(entry, dict)
        and entry.get("path") == path
        and entry.get("version") == version
    ]
    if len(matches) != 1:
        raise ValueError(f"No unique official compiler build for {version}")
    entry = matches[0]
    if not isinstance(entry.get("sha256"), str):
        raise ValueError(f"Official build lacks SHA-256 for {version}")
    return entry


def qualify(
    output_dir: Path, versions: list[str], *, resume: bool = False
) -> dict[str, object]:
    if output_dir.is_symlink():
        raise ValueError("Output directory cannot be a symlink")
    if output_dir.exists() and not resume:
        raise FileExistsError("Output directory must be new unless --resume is set")
    if not versions or len(versions) > 4 or len(set(versions)) != len(versions):
        raise ValueError("Request one to four distinct compiler versions")
    if any(VERSION_RE.fullmatch(version) is None for version in versions):
        raise ValueError("Invalid compiler version")
    output_dir.mkdir(parents=True, exist_ok=resume)
    receipts: list[dict[str, object]] = []
    list_path = output_dir / "list.json"
    if list_path.exists():
        if not resume or list_path.is_symlink() or list_path.stat().st_size > MAX_LIST_BYTES:
            raise FileExistsError(list_path)
        receipts.append(
            {
                "url": BASE_URL + "list.json",
                "bytes": list_path.stat().st_size,
                "sha256": _sha256(list_path),
                "resumed_existing": True,
            }
        )
    else:
        receipts.append(_fetch(BASE_URL + "list.json", list_path, MAX_LIST_BYTES))
    listing = json.loads(list_path.read_text())
    compilers: list[dict[str, object]] = []
    for version in versions:
        entry = _select_build(listing, version)
        build_path = entry["path"]
        if not isinstance(build_path, str) or "/" in build_path or "\\" in build_path:
            raise ValueError("Unsafe compiler build path")
        binary = output_dir / f"solc-{version}"
        published = str(entry["sha256"]).removeprefix("0x").lower()
        if binary.exists():
            if (
                not resume
                or binary.is_symlink()
                or binary.stat().st_size > MAX_BINARY_BYTES
                or _sha256(binary) != published
            ):
                raise FileExistsError(f"Existing compiler is not resumable: {binary}")
            receipt = {
                "url": BASE_URL + build_path,
                "bytes": binary.stat().st_size,
                "sha256": published,
                "resumed_existing": True,
            }
        else:
            receipt = _fetch(BASE_URL + build_path, binary, MAX_BINARY_BYTES)
        receipts.append(receipt)
        if receipt["sha256"] != published:
            raise ValueError(f"SHA-256 mismatch for compiler {version}")
        binary.chmod(
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IROTH
        )
        process = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if process.returncode or f"Version: {version}+" not in process.stdout:
            raise RuntimeError(f"Verified binary reported unexpected version: {version}")
        compilers.append(
            {
                "version": version,
                "path": binary.name,
                "bytes": binary.stat().st_size,
                "sha256": published,
                "official_build": entry,
                "version_stdout": process.stdout,
            }
        )
    result = {
        "status": "passed_official_checksum_and_version_qualification",
        "platform": "macosx-amd64",
        "official_list_url": BASE_URL + "list.json",
        "official_list_sha256": _sha256(list_path),
        "compilers": compilers,
        "receipts": receipts,
        "global_install": False,
        "qualified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / "qualification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--versions", required=True, nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    lock = Path("/tmp/paper-research-compute.lock")
    with lock.open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Shared paper compute lock is held") from error
        result = qualify(args.output_dir, args.versions, resume=args.resume)
        fcntl.flock(lease, fcntl.LOCK_UN)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
