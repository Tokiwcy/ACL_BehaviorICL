"""Reuse frozen v4 training/retrieval functions with a separate source binding."""
import run_internvl35_qwen_methods_v4 as base
from internvl35_joint_config_v1 import DATASETS, METHODS, MODEL, OUTPUT, REVISION, PROTOCOL_PATH, digest, load_model, write_json, ensure_protocol, require_gate
for name in ("DATASETS", "OUTPUT", "PROTOCOL_PATH", "load_model", "write_json", "ensure_protocol"):
    setattr(base, name, globals()[name])
load_context = base.load_context
original_require_anchor = base.require_anchor

def require_anchor(folder, total):
    import json
    require_gate()
    progress = json.loads((folder / "joint_progress.json").read_text(encoding="utf-8"))
    receipt = json.loads((folder / "joint_binding.json").read_text(encoding="utf-8"))
    if progress["completed"] != total or progress["binding"] != receipt or receipt["protocol_sha256"] != digest(PROTOCOL_PATH):
        raise ValueError("Joint source commit or protocol incomplete")
    return original_require_anchor(folder, total)

base.require_anchor = require_anchor
if __name__ == "__main__":
    base.main()
