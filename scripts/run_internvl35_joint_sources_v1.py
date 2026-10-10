"""Production source stage; compatibility gate required before CUDA loading."""
import argparse
import json
from internvl35_joint_config_v1 import *
from run_internvl35_joint_methods_v1 import load_context

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--stage", choices=("joint", "clip"), default="joint")
    args = parser.parse_args()
    require_gate()
    folder, bank, query, identity = load_context(args.dataset)
    from run_internvl35_qwen_sources_v4 import freeze_manifest
    freeze_manifest(args.dataset, bank, query, folder)
    if args.stage == "clip":
        import run_multidataset_main as main_run
        main_run.write_json = write_json
        main_run.extract_clip_cache(bank + query, ROOT / ".hf_cache", 32, folder / "clip_features.npy", True)
        return
    from internvl35_joint_source_v1 import extract_dataset
    binding = {"protocol_sha256": digest(PROTOCOL_PATH), "manifest_sha256": identity["manifest_sha256"],
               "compatibility_sha256": digest(ANALYSIS / "compatibility.json")}
    receipt = folder / "joint_binding.json"
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) != binding:
        raise ValueError("Source compatibility receipt changed")
    write_json(receipt, binding)
    model, processor = load_model(folder / "telemetry/joint_source.jsonl")
    extract_dataset(model, processor, bank, query, args.dataset, sorted({s.label for s in bank}), folder, binding)

if __name__ == "__main__":
    main()
