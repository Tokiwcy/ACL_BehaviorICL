"""Offline verification of the October 10 published local supplement."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'handoff/local_publish_manifest_20261010.json'
COLLABORATOR_MANIFEST = ROOT / 'handoff/local_publish_manifest_20261010_collaborator_v1.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='Verify a specific immutable publication snapshot')
    args = parser.parse_args()
    selected = args.manifest or (COLLABORATOR_MANIFEST if COLLABORATOR_MANIFEST.exists() else MANIFEST)
    manifest = json.loads(selected.read_text(encoding='utf-8'))
    if 'previous_manifest_sha256' in manifest:
        if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != manifest['previous_manifest_sha256']:
            raise ValueError('Previous publication manifest changed')
    seen = set()
    secret_patterns = [rb'\bgh[oprsu]_[A-Za-z0-9]{20,}\b',
                       rb'\bgithub_pat_[A-Za-z0-9_]{30,}\b',
                       rb'\bhf_[A-Za-z0-9]{24,}\b', rb'\bAKIA[A-Z0-9]{16}\b',
                       rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                       rb'\brpa_[A-Za-z0-9]{20,}\b', rb'\bsk-[A-Za-z0-9_-]{20,}\b']
    for entry in manifest['files']:
        rel = PurePosixPath(entry['path'])
        if rel.is_absolute() or '..' in rel.parts or entry['path'] in seen:
            raise ValueError('Invalid or duplicate publication path')
        if rel.parts[0] in {'results', 'datasets', '.tools', '.hf_cache', '.venv'}:
            raise ValueError('Private input directory published')
        if rel.suffix in {'.tmp', '.npy', '.npz', '.pt', '.pth', '.safetensors', '.pem', '.key'}:
            raise ValueError('Weight, cache or incomplete file published')
        seen.add(entry['path'])
        data = (ROOT / rel).read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f"Published bytes differ: {rel}; use the published commit for this snapshot")
        if any(re.search(pattern, data) for pattern in secret_patterns):
            raise ValueError(f'Credential-like content: {rel}')
        if rel.suffix == '.py':
            ast.parse(data, filename=str(rel))
        elif rel.suffix == '.json':
            json.loads(data)
    audit = json.loads((ROOT / 'analysis/internvl35_joint_v2/combined_results_audit.json').read_text())
    completed = sum(row['state'] == 'complete' for dataset in audit['datasets'].values()
                    for row in dataset['methods'].values())
    if completed != manifest['dynamic_full_test_audited_cells'] or audit['state'] != 'incomplete':
        raise ValueError('Full-test completion scope differs from publication snapshot')
    if len(audit['incomplete']) != 24 - completed:
        raise ValueError('Composite pending list does not match merged method states')
    if hashlib.sha256((ROOT / 'handoff/publish_manifest.json').read_bytes()).hexdigest() != manifest['original_manifest_sha256']:
        raise ValueError('Original handoff manifest changed')
    print(json.dumps({'state': 'local_publication_verified', 'files': len(seen),
                      'publication_version': manifest['version'],
                      'dynamic_full_test_audited_cells': completed, 'dynamic_total_cells': 24,
                      'dtd_old_single_patch_audited_cells': 6,
                      'scientific_model_replay': False, 'live_gpu_state_checked': False}, indent=2))


if __name__ == '__main__':
    main()
