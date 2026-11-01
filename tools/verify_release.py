"""Verify the curated package, manuscript evidence and paper-wide coverage offline."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'provenance/release_manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    ignored = {'.git','.pytest_cache','__pycache__','build','dist'}
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*')
            if p.is_file() and p != MANIFEST and not any(
                part in ignored or part.startswith(('.pytest_tmp','.venv')) or part.endswith('.egg-info')
                for part in p.relative_to(ROOT).parts)}


def rows(name):
    with (ROOT/'results/paper'/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--write-manifest',action='store_true',help='maintainer: bind an intentionally reviewed tree')
    args=parser.parse_args()
    if args.write_manifest:
        MANIFEST.write_text(json.dumps({'format':'decisionicl-core-release-v1','files':files()},indent=2),encoding='utf-8')
    declared=json.loads(MANIFEST.read_text(encoding='utf-8'))['files']
    actual=files()
    assert declared == actual, 'release tree differs: '+str(sorted(set(declared)^set(actual)) or [p for p in declared if declared[p]!=actual[p]])
    records=json.loads((ROOT/'provenance/source_manifest.json').read_text(encoding='utf-8'))
    for record in records:
        assert sha(ROOT/record['destination']) == record['published_sha256'], record['destination']
    for path in (ROOT/'src').rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    # Every supplement section remains present in the unchanged inlined source.
    supplement=(ROOT/'paper/supplement.tex').read_text(encoding='utf-8')
    sections=re.findall(r'\\section\{([^}]+)\}',supplement)
    assert len(sections)==11, sections
    for entry in ('main.tex','supplement.tex'):
        text=(ROOT/'paper'/entry).read_text(encoding='utf-8')
        for figure in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
            assert (ROOT/'paper'/figure).is_file(), figure
    assert len(list((ROOT/'results/tables').glob('*.tex')))==28
    assert len(rows('main_method_results.csv'))==214
    assert len(rows('additional_baseline_results.csv'))==90
    assert len(rows('additional_baseline_pairs.csv'))==60
    repeats=rows('three_seed_method_results.csv')
    assert len(repeats)==180
    assert len(rows('shot_curve_results.csv'))==12
    bank_index=json.loads((ROOT/'resources/factor_banks/index.json').read_text(encoding='utf-8'))
    assert len(bank_index)==15
    for bank in bank_index:
        assert sha(ROOT/bank['path'])==bank['sha256']
    assert not (ROOT/'scripts').exists() or not list((ROOT/'scripts').glob('*.py'))
    print(json.dumps({'status':'passed','files':len(actual),'supplement_sections':len(sections),
        'numerical_tables':28,'main_cells':180,'auxiliary_cells':90,'auxiliary_pairs':60,
        'three_run_cells':len(repeats),'higher_shot_cells':12,'frozen_main_banks':15,
        'target_model_inference':False},sort_keys=True))


if __name__=='__main__':
    main()
