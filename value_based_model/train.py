"""Train DRE/NDRE visit-status forests from a prepared 3/6/9-month bundle."""
import argparse
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path,
                        help='Override the default results/rf_visit_status_YYYYMMDD_HHMMSS_<trees>trees folder (Asia/Taipei time)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--panel', choices=['core', 'all'], default='all')
    parser.add_argument('--top-features', type=int, default=20)
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    stamp = dt.datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S')
    output = args.output_dir or ROOT / 'results' / f'rf_visit_status_{stamp}_{args.trees}trees'
    from visit_status_workflow import run_status_training
    run_status_training(args.feature_dir, output, seed=args.seed, trees=args.trees, jobs=args.jobs,
                        bootstrap=args.bootstrap, top_features=args.top_features, panel=args.panel,
                        progress=lambda text: print(text, flush=True))
    print('Saved visit-status results: ' + str(output.resolve()), flush=True)


if __name__ == '__main__':
    main()
