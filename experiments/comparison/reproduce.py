"""Run the complete fixed-data comparison evaluation in a fresh checkout."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/comparison'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if (ROOT / 'results/comparison-panel').exists():
        raise FileExistsError('Use a fresh checkout; existing comparison outputs are preserved')
    for name in ['results', 'records', 'figures', 'tables']:
        (ROOT / name).mkdir(exist_ok=True)
    env = os.environ.copy()
    # Keep any externally installed isolation hook, while making imports explicit.
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT/'src'), str(EXPERIMENT), env.get('PYTHONPATH')]))
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
               XDG_CACHE_HOME=str(ROOT/'cache'))
    steps = []
    started = time.perf_counter()

    def run(name, arguments):
        start = time.perf_counter()
        with (ROOT / 'records' / (name + '.log')).open('x') as stream:
            result = subprocess.run([sys.executable, '-B', *arguments], cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        steps.append(dict(name=name, arguments=arguments, returncode=result.returncode,
                          seconds=time.perf_counter()-start))
        (ROOT/'records/steps.json').write_text(json.dumps(steps, indent=2)+'\n')
        print(name, 'return', result.returncode, 'seconds', round(steps[-1]['seconds'], 2), flush=True)
        if result.returncode:
            raise RuntimeError('Failed stage: ' + name + '; inspect records/' + name + '.log')

    prefix = 'experiments/comparison/'
    run('inputs', [prefix+'acquire.py', '--verify'])
    run('core-tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    run('comparison-tests', ['-m', 'unittest', 'discover', '-s', 'tests/comparison', '-v'])
    for name in ['comparison_panel', 'verify_comparison_panel', 'comparison_robustness',
                 'verify_comparison_robustness', 'comparison_ranking', 'verify_comparison_ranking']:
        run(name, [prefix+name+'.py'])
    run('expanded-selection', [prefix+'comparison_ranking_boundary.py', '--stage', 'select'])
    run('verify-selection', [prefix+'verify_ranking_boundary_selection.py'])
    selection = hashlib.sha256((ROOT/'results/comparison-ranking-boundary/selection-manifest.json').read_bytes()).hexdigest()
    run('expanded-evaluation', [prefix+'comparison_ranking_boundary.py', '--stage', 'evaluate', '--expected-selections', selection])
    for name in ['verify_ranking_boundary_results', 'comparison_mechanisms',
                 'verify_comparison_accounting', 'verify_comparison_ranks',
                 'calibration_scope', 'verify_calibration_scope', 'check_results', 'render_results']:
        run(name, [prefix+name+'.py'])
    packages = {name:importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'matplotlib']}
    record = dict(status='passed', seconds=time.perf_counter()-started, steps=steps,
                  python=platform.python_version(), platform=platform.platform(), packages=packages,
                  scope='Fresh fitting, selection, scoring and rendering on five exact retained CSVs; development-data reproduction, not independent scientific validation.')
    (ROOT/'records/reproduction.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'status':'passed', 'seconds':record['seconds']}), flush=True)


if __name__ == '__main__':
    main()
