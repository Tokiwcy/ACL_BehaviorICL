"""Write the independent CPU audit after the local full GPU supervisor finishes."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results/internvl35_full_v1/pipeline_logs/status.json"
LOG = ROOT / "results/internvl35_full_v1/pipeline_logs/final_audit.log"


def main() -> None:
    while True:
        status = json.loads(STATUS.read_text(encoding="utf-8"))
        if status["state"] == "complete":
            break
        if status["state"] == "failed":
            LOG.write_text("GPU pipeline failed; no final audit was produced.\n", encoding="utf-8")
            raise SystemExit(1)
        time.sleep(60)
    commands = [
        [sys.executable, str(ROOT / "scripts/audit_internvl35_full_v1.py")],
        [sys.executable, str(ROOT / "scripts/verify_handoff.py")],
    ]
    with LOG.open("w", encoding="utf-8") as handle:
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
