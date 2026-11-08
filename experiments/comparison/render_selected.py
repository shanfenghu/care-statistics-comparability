"""Render the selected scientific displays from recorded summaries, without fitting.

Write to a new directory so the distributed figures, tables and results remain
unchanged. No provider inputs or dense case records are needed for these displays.
"""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ('comparison-panel', 'comparison-robustness', 'comparison-ranking-stable',
          'comparison-ranking-boundary', 'comparison-mechanism', 'calibration-scope')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'selected/results',
                        help='Directory containing the six recorded summary groups')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for regenerated displays and supporting summaries')
    args = parser.parse_args()
    members = [Path(group)/'summary.json' for group in GROUPS]
    members.append(Path('comparison-robustness/verification.json'))
    for member in members:
        source = args.results_dir/member
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError('Missing ordinary result file: ' + str(source))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for member in members:
        destination = output/'results'/member
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.results_dir/member, destination)
    for name in ('figures', 'tables', 'records'):
        (output/name).mkdir()
    import render_results
    import render_cutoff
    import export_tables
    # Existing renderers read these summaries; the estimation code is not run.
    render_results.ROOT = output
    render_cutoff.VENUE = output
    export_tables.ROOT = output
    render_results.main()
    print('Rendered selected displays without fitting in', output)


if __name__ == '__main__':
    main()
