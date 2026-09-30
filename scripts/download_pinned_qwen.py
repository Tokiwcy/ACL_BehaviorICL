"""Populate the Qwen cache only if the public main ref matches the run's pinned commit."""

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
EXPECTED_COMMIT = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def main() -> None:
    current = HfApi().model_info(MODEL_ID).sha
    if current != EXPECTED_COMMIT:
        raise RuntimeError(f"Model main commit changed: {current}; expected {EXPECTED_COMMIT}")
    snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision="main"))
    if snapshot.name != EXPECTED_COMMIT:
        raise RuntimeError(f"Downloaded unexpected model commit: {snapshot.name}")
    print(f"Model snapshot ready: {snapshot.name}", flush=True)


if __name__ == "__main__":
    main()
