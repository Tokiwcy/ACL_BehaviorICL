"""Restore exact frozen JSON/CSV payloads offline; never overwrite conflicts."""
from __future__ import annotations

import os
from handoff_artifact_utils import manifest, repository_path, sha256, verified_payload


def main() -> None:
    restored = existing = 0
    entries = [entry for entry in manifest()["files"] if entry["kind"] == "gzip_exact"]
    # Validate every input and every existing destination before the first write.
    for entry in entries:
        verified_payload(entry)
        destination = repository_path(entry["materialize_path"])
        if destination.exists() and (
            destination.stat().st_size != entry["source_bytes"]
            or sha256(destination.read_bytes()) != entry["source_sha256"]
        ):
            raise ValueError(f"Conflicting existing file; preserved: {entry['materialize_path']}")
        if destination.with_name(destination.name + ".handoff-part").exists():
            raise ValueError(f"Existing partial file; inspect before retry: {entry['materialize_path']}")
    for entry in entries:
        destination = repository_path(entry["materialize_path"])
        if destination.exists():
            existing += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".handoff-part")
        with temporary.open("xb") as stream:
            stream.write(verified_payload(entry))
            stream.flush()
            os.fsync(stream.fileno())
        if sha256(temporary.read_bytes()) != entry["source_sha256"]:
            raise ValueError(f"Written SHA mismatch: {entry['materialize_path']}")
        if destination.exists():
            raise ValueError(f"Destination appeared during materialization: {entry['materialize_path']}")
        # Atomic creation refuses a destination introduced by a concurrent writer.
        os.link(temporary, destination)
        temporary.unlink()
        restored += 1
    print(f"Offline artifacts verified: {len(entries)}; restored: {restored}; already exact: {existing}.")
    print("No image, weight, model, or cloud operation performed.")


if __name__ == "__main__":
    main()
