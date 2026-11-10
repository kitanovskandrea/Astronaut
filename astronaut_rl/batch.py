"""Run independent curriculum chains in separate Python processes.

Stages within a chain remain sequential. Timings are active wall time under
concurrent load. Outputs are merged without changing model bytes or records.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .config import TRAINING_SEEDS
from .experiment import _write_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('parallel_results'))
    parser.add_argument('--models', type=Path, default=Path('parallel_models'))
    parser.add_argument('--work', type=Path, default=Path('parallel_jobs'))
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(TRAINING_SEEDS))
    args = parser.parse_args()
    if not 1 <= args.workers <= len(args.seeds) or len(args.seeds) != len(set(args.seeds)):
        parser.error('Use a positive worker count and distinct training seeds')
    for path in (args.out, args.models, args.work):
        if path.exists() and any(path.iterdir()):
            parser.error('Choose an empty output directory: ' + str(path))
        path.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    pending, active, completed = list(args.seeds), {}, {}
    project = Path(__file__).resolve().parents[1]
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                seed = pending.pop(0)
                job = args.work.resolve() / ('seed' + str(seed))
                job.mkdir()
                with (job / 'training.log').open('w') as log:
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'astronaut_rl.experiment',
                        '--seeds', str(seed), '--out', str(job / 'results'),
                        '--models', str(job / 'models'), '--progress', str(job / 'progress.json')],
                        cwd=project, stdout=log, stderr=subprocess.STDOUT)
                active[seed] = (process, job)
                print('START chain', seed, flush=True)
            for seed, (process, job) in list(active.items()):
                if process.poll() is None:
                    continue
                if process.returncode not in (0, 1) or not (job / 'results/run_manifest.json').exists():
                    raise RuntimeError('Chain failed unexpectedly; inspect ' + str(job / 'training.log'))
                manifest = json.loads((job / 'results/run_manifest.json').read_text())
                if not manifest['complete']:
                    raise RuntimeError('Incomplete chain: ' + str(seed))
                completed[seed] = (manifest, job)
                del active[seed]
                print('FINISH chain', seed, 'qualified=', manifest['curriculum_passed'], flush=True)
            if active:
                time.sleep(.2)
    finally:
        for process, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, _ in active.values():
            process.wait()
    first = completed[args.seeds[0]][0]
    combined = {**first, 'created_utc': created, 'training_seeds': args.seeds,
        'records': [], 'failed_stages': [], 'parallel_chains': args.workers,
        'batch_elapsed_seconds': time.perf_counter() - start,
        'timing_protocol': 'independent parallel chains; sequential stages within each chain; active per-process wall time under concurrent load excludes initialization, validation, test, export and progress IO'}
    for seed in args.seeds:
        manifest, job = completed[seed]
        combined['records'].extend(manifest['records'])
        combined['failed_stages'].extend(manifest['failed_stages'])
        for name, destination in (('models', args.models), ('results', args.out)):
            for source in (job / name).iterdir():
                if source.name == 'run_manifest.json':
                    continue
                target = destination / source.name
                if source.is_dir():
                    shutil.copytree(source, target, dirs_exist_ok=True)
                else:
                    if target.exists():
                        raise RuntimeError('Duplicate batch artifact: ' + str(target))
                    shutil.copy2(source, target)
    combined['curriculum_passed'] = not combined['failed_stages']
    _write_progress(args.out / 'run_manifest.json', combined)
    print('Batch elapsed seconds:', round(combined['batch_elapsed_seconds'], 2), flush=True)
    raise SystemExit(0 if combined['curriculum_passed'] else 1)


if __name__ == '__main__':
    main()
