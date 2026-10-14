"""Prepare an isolated run directory and call the unchanged research scripts."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    'theory': ('theory_hub.py', '.', []),
    'theory-checks': ('theory_checks.py', '.', []),
    'synthetic-PBF': ('exp_rev.py', 'res/rev', ['PBF']),
    'synthetic-CHRASE': ('exp_rev.py', 'res/rev2/code', ['CHRASE']),
    'synthetic-QM': ('exp_rev.py', 'res/rev3/code', ['QM']),
    'synthetic-H': ('exp_rev.py', 'res/rev4/code', ['H']),
    'empirical': ('exp_emp_rev.py', 'res/rev3/code', ['SXDLHV']),
    'eventstudy': ('emp_eventstudy_rev.py', '.', []),
    'analyze': ('analyze_rev.py', '.', []),
    'analyze-empirical': ('analyze_emp_rev.py', '.', []),
    'extract-panels': ('build_panel.py', '.', ['extract']),
}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or digest(source) != digest(target):
            raise FileExistsError(f'Refusing to replace a different file: {target}')
        return
    with source.open('rb') as reader, target.open('xb') as writer:
        shutil.copyfileobj(reader, writer)
    if digest(source) != digest(target):
        raise RuntimeError(f'Copy verification failed: {target}')

def copy_tree(source, target):
    if not source.is_dir():
        return
    for item in sorted(source.rglob('*')):
        if item.is_symlink():
            raise RuntimeError(f'Symbolic links are not supported: {item}')
        if item.is_file():
            copy_file(item, target / item.relative_to(source))

def check():
    scripts = sorted((ROOT / 'code').glob('*.py'))
    for script in scripts:
        ast.parse(script.read_text(encoding='utf-8-sig'), filename=str(script))
    required = ['numpy', 'scipy', 'pandas', 'numba', 'statsmodels', 'pyarrow']
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    print(f'Parsed {len(scripts)} research scripts. Processed panels: {len(list((ROOT / "data" / "panels").glob("emp_*.npz")))}')
    if missing:
        print('Missing installed Python dependencies: ' + ', '.join(missing))
        return 1
    print('Required Python packages are installed. No experiment has been started.')
    return 0

def prepare(work, raw_dir):
    if work == ROOT or work.is_relative_to(ROOT):
        raise ValueError('Choose a run directory outside this source bundle.')
    if work.exists():
        raise FileExistsError(f'Choose a new run directory; this one already exists: {work}')
    work.mkdir(parents=True, exist_ok=False)
    copy_tree(ROOT / 'code', work / 'code')
    copy_tree(ROOT / 'data' / 'summaries', work / 'handoff')
    copy_tree(ROOT / 'data' / 'panels', work / 'handoff')
    copy_tree(ROOT / 'data' / 'panels', work / 'panels')
    copy_tree(ROOT / 'data' / 'panels', work / 'res/rev3/code/panels')
    for group, location in {
        'synthetic_PBF': 'res/rev/out_rev',
        'synthetic_CHRASE': 'res/rev2/code/out_rev',
        'synthetic_QM': 'res/rev3/code/out_rev',
        'synthetic_H_rerun': 'res/rev4/code/out_rev',
        'empirical_model': 'res/rev3/code/out_emp_rev',
        'empirical_eventstudy': 'res/emp_rev',
    }.items():
        copy_tree(ROOT / 'data' / 'results' / group, work / location)
    copy_tree(ROOT / 'figures', work / 'figs_rev')
    copy_tree(ROOT / 'build_inputs' / 'manuscript_sources', work / 'ms_rev')
    copy_tree(ROOT / 'build_inputs' / 'latex_dependencies', work / 'ms_rev')
    if raw_dir:
        raw = Path(raw_dir).resolve(strict=True)
        if not raw.is_dir():
            raise ValueError('--raw-dir must be a directory')
        for file in sorted(raw.iterdir()):
            if file.is_file() and (file.name in {'tm.zip','qcl.zip','country_names.csv'} or file.suffix == '.parquet'):
                copy_file(file, work / 'data_raw' / file.name)
    marker = {'source_bundle': str(ROOT), 'note': 'Inputs and scripts copied; no experiments run.'}
    (work / 'RUN_ORIGIN.json').write_text(json.dumps(marker, indent=2), encoding='utf-8')
    print(f'Prepared and hash-checked: {work}')

def run_stage(stage, work, nproc):
    if not (work / 'RUN_ORIGIN.json').is_file():
        raise ValueError('Prepare a new run directory first.')
    marker = json.loads((work / 'RUN_ORIGIN.json').read_text(encoding='utf-8'))
    if Path(marker['source_bundle']).resolve() != ROOT:
        raise ValueError('This run directory belongs to another source bundle.')
    script, relative_cwd, blocks = STAGES[stage]
    cwd = work / relative_cwd
    cwd.mkdir(parents=True, exist_ok=True)
    if stage == 'extract-panels':
        required = [work / 'data_raw' / name for name in ('tm.zip', 'qcl.zip')]
        absent = [str(p) for p in required if not p.is_file()]
        if absent:
            raise FileNotFoundError('Missing FAOSTAT raw files: ' + ', '.join(absent))
    command = [sys.executable, str(work / 'code' / script)]
    if script in {'exp_rev.py', 'exp_emp_rev.py'}:
        command += [str(nproc)]
    command += blocks
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    print(f'Running {stage} in {cwd}', flush=True)
    rc = subprocess.call(command, cwd=cwd, env=env)
    if rc == 0 and stage == 'extract-panels':
        # Synchronize newly generated panels only within the disposable run directory.
        for file in sorted((work / 'handoff').glob('emp_*.npz')):
            for dest in (work / 'panels', work / 'res/rev3/code/panels'):
                shutil.copy2(file, dest / file.name)
    return rc

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'prepare'] + list(STAGES))
    parser.add_argument('--work-dir', help='New isolated directory for preparation, or an existing prepared directory for execution')
    parser.add_argument('--raw-dir', help='Optional local directory containing tm.zip, qcl.zip or required Parquet inputs; no downloads')
    parser.add_argument('--nproc', type=int, default=1, help='Worker count passed unchanged to the original experiment script')
    args = parser.parse_args()
    if args.nproc < 1:
        parser.error('--nproc must be positive')
    if args.stage == 'check':
        return check()
    if not args.work_dir:
        parser.error('--work-dir is required')
    work = Path(args.work_dir).resolve()
    if args.stage == 'prepare':
        prepare(work, args.raw_dir)
        return 0
    if args.raw_dir:
        parser.error('--raw-dir is used only with prepare')
    return run_stage(args.stage, work, args.nproc)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(2)
