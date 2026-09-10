"""Print batch progress using only the standard library; does not start Isaac."""
import argparse
import json
from pathlib import Path
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', nargs='?', type=Path,
               default=Path(__file__).with_name('batches') / 'ten_20260905_a')
args = p.parse_args()
state = json.loads((args.directory / 'status.json').read_text())
print(f"Batch: {state['status']} | completed {state['completed']}/{len(state['trials'])}")
for trial in state['trials']:
    done = trial.get('last_iteration', -1) + 1
    print(f"{trial['index']:02d} {trial['name']:20s} {trial['status']:12s} "
          f"{done:3d}/{trial['iterations']} updates | {trial.get('elapsed_s', 0)}s")
    if 'log_dir' in trial:
        print(f"   {trial['log_dir']}")
print(f"Status age: {time.time() - (args.directory / 'status.json').stat().st_mtime:.0f}s")
print('Metrics describe training rollouts; they are not held-out 15 cm stair success rates.')
