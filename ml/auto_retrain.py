#!/usr/bin/env python3
"""
Unattended retrain-if-improved loop -- safe to run on a schedule (cron/launchd).

This only automates the mechanical parts of the training loop (downloading
new scans, retraining, rebuilding the index, evaluating). It never touches
identity confirmation: label_flashes.py stays a manual, human-reviewed step,
since at current accuracy auto-accepting the model's own guesses would inject
wrong labels often enough to actively hurt the model (see eval numbers).

Each run:
  1. python3 download_flashes.py     -- pull in new scans (unattended-safe)
  2. Check: are there >= --min-new new confirmed labels since the last
     retrain? If not, stop here -- nothing to do yet.
  3. Back up clip_finetuned.pt + embeddings_clip.npz + eval_latest.json
  4. python3 train_clip.py
  5. python3 extract_features_clip.py
  6. python3 eval_recognition.py
  7. Compare the new multi-crop top-1 against the pre-retrain eval_latest.json.
     Improved (or no prior baseline yet) -> keep the new checkpoint, advance
     retrain_state.json. Regressed -> restore the backup, leave
     retrain_state.json alone so it retries once more labels accumulate.

Usage (from ml/ directory):
  python3 auto_retrain.py                # default threshold (300 new confirms)
  python3 auto_retrain.py --min-new 500
  python3 auto_retrain.py --dry-run      # report what it would do, no changes
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from flash_split import is_eval

IDENTITY_LABELS_FILE = Path('flashes/identity_labels.json')
STATE_FILE           = Path('retrain_state.json')
MODEL_FILE           = Path('clip_finetuned.pt')
EMB_FILE             = Path('embeddings_clip.npz')
LATEST_EVAL_FILE     = Path('eval_latest.json')

DEFAULT_MIN_NEW = 300

def count_confirmed():
    try:
        labels = json.loads(IDENTITY_LABELS_FILE.read_text())
    except Exception:
        return 0
    return sum(1 for l in labels.values() if l.get('verdict') == 'confirmed' and l.get('poi_id'))

def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {'last_retrain_confirmed_count': 0, 'last_retrain_time': None}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def load_eval(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def run(*args):
    print(f'\n$ python3 {" ".join(args)}')
    subprocess.run([sys.executable, *args], check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--min-new', type=int, default=DEFAULT_MIN_NEW,
                         help=f'minimum new confirmed labels since last retrain to trigger one (default: {DEFAULT_MIN_NEW})')
    parser.add_argument('--dry-run', action='store_true', help="report what would happen, don't change anything")
    parser.add_argument('--skip-download', action='store_true', help='skip the download_flashes.py step')
    args = parser.parse_args()

    if not args.skip_download and not args.dry_run:
        run('download_flashes.py')

    state = load_state()
    current_confirmed = count_confirmed()
    new_since_last = current_confirmed - state['last_retrain_confirmed_count']

    print(f'\nConfirmed labels: {current_confirmed} total, '
          f'{new_since_last} new since last retrain '
          f'({state["last_retrain_time"] or "never"})')

    if new_since_last < args.min_new:
        print(f'Below threshold ({args.min_new}) -- nothing to do.')
        return

    print(f'Threshold met ({new_since_last} >= {args.min_new}).')
    if args.dry_run:
        print('--dry-run: would retrain now.')
        return

    prev_eval = load_eval(LATEST_EVAL_FILE)

    backups = {}
    for f in (MODEL_FILE, EMB_FILE, LATEST_EVAL_FILE):
        if f.exists():
            bak = f.with_suffix(f.suffix + '.bak')
            shutil.copy(f, bak)
            backups[f] = bak

    run('train_clip.py')
    run('extract_features_clip.py')
    run('eval_recognition.py')

    new_eval = load_eval(LATEST_EVAL_FILE)
    if new_eval is None:
        print('No eval result produced (no held-out examples?) -- keeping new checkpoint anyway.')
        improved = True
    elif prev_eval is None:
        print('No prior baseline to compare against -- keeping new checkpoint.')
        improved = True
    else:
        prev_top1 = prev_eval['multi_crop']['top1']
        new_top1  = new_eval['multi_crop']['top1']
        improved = new_top1 >= prev_top1
        print(f'multi-crop top-1: {prev_top1:.1%} -> {new_top1:.1%}  '
              f'({"improved" if improved else "regressed"})')

    if improved:
        state['last_retrain_confirmed_count'] = current_confirmed
        state['last_retrain_time'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        save_state(state)
        print('Kept new checkpoint.')
    else:
        for f, bak in backups.items():
            shutil.copy(bak, f)
        print('Reverted to previous checkpoint (retrain_state.json left as-is, will retry with more labels).')

if __name__ == '__main__':
    main()
