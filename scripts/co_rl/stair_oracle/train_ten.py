"""Sequential, bounded PPO comparison. Run through run_train_ten.sh."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.co_rl.stair_oracle.replay_guard import acquire_replay_lock, available_memory_kib

FLAT = ROOT / 'logs/co_rl/Flamingo_Light_Flat_Stand_Drive/ppo/2026-07-02_12-12-49/model_1300.pt'
LOGS = ROOT / 'logs/co_rl/Flamingo_Light_Rough_Stair_Jump/ppo'
VARIANTS = [
    ('baseline', {}),
    ('fixed_lr_3e4', {'schedule': 'fixed', 'learning_rate': 0.0003}),
    ('fixed_lr_1e4', {'schedule': 'fixed', 'learning_rate': 0.0001}),
    ('low_entropy', {'entropy_coef': 0.003}),
    ('high_entropy', {'entropy_coef': 0.02}),
    ('tight_clip', {'clip_param': 0.1}),
    ('tight_kl', {'desired_kl': 0.005}),
    ('long_horizon', {'gamma': 0.995}),
    ('long_gae', {'lam': 0.98}),
    ('fewer_epochs', {'num_learning_epochs': 3}),
]


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def metrics(log_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(log_dir), size_guidance={'scalars': 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()['scalars']:
        if any(s in tag for s in ('terrain_levels', 'climb_units', 'mean_reward', 'mean_episode_length')):
            vals = ea.Scalars(tag)[-50:]
            if vals:
                out[tag] = sum(v.value for v in vals) / len(vals)
    attempts = ROOT / 'scripts/co_rl/stair_oracle/data' / log_dir.name / 'stair_attempts.csv'
    if attempts.exists():
        with attempts.open() as f:
            rows = list(csv.DictReader(f))
        out['training_attempt_count'] = len(rows)
        out['training_attempt_success_count'] = sum(int(r['success']) for r in rows)
        out['training_attempt_success_rate'] = out['training_attempt_success_count'] / len(rows) if rows else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not FLAT.is_file():
        raise FileNotFoundError(FLAT)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise RuntimeError('Existing batch status found; use a new output directory to avoid duplicate runs')
    base = json.loads((ROOT / 'scripts/co_rl/sweeps/rough_warmstart_flat1300_5k.json').read_text())
    base.update({
        'env.commands.base_velocity.ranges.lin_vel_x': [-0.8, 0.8],
        'env.commands.base_velocity.ranges.ang_vel_z': [-0.6, 0.6],
        'agent.save_interval': 100,
    })
    plan = []
    for index, (name, changes) in enumerate(VARIANTS, 1):
        overrides = {**base, **{'agent.algorithm.' + k: v for k, v in changes.items()}}
        spec = out / f'{index:02d}_{name}.json'
        save(spec, overrides)
        run_name = f'{out.name}_t{index:02d}_{name}'
        cmd = [sys.executable, '-u', 'scripts/co_rl/train.py', '--task',
               'Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-v1-ppo', '--algo', 'ppo',
               '--headless', '--batch_exit', '--num_envs', '1024', '--seed', '42',
               '--max_iterations', '300', '--run_name', run_name,
               '--warmstart_ckpt', str(FLAT), '--param_overrides', str(spec)]
        plan.append({'index': index, 'name': name, 'run_name': run_name, 'command': cmd,
                     'status': 'pending', 'iterations': 300})
    save(out / 'manifest.json', {'trials': plan, 'total_update_budget': 3000,
         'source': str(FLAT), 'num_envs': 1024, 'seed': 42,
         'reward_changes': False, 'max_trial_seconds': 1200,
         'note': 'Training metrics are preliminary, not held-out evaluation or BC success.'})
    if args.dry_run:
        print(f'Validated {len(plan)} trials: {out / "manifest.json"}')
        return
    lock = acquire_replay_lock(ROOT / 'scripts/co_rl/stair_oracle/waypoint_runs/.replay.lock')
    state = {'status': 'running', 'completed': 0, 'trials': plan}
    child = None

    def stop_child():
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for trial in plan:
            if available_memory_kib() < 6 * 1024 * 1024:
                state['status'] = 'stopped_low_memory'
                break
            # Refuse overlap with a simulator started outside this batch.
            listing = subprocess.check_output(['ps', '-eo', 'comm,args'], text=True)
            if any(line.split()[0].startswith('python') and any(s in line for s in
                   ('scripts/co_rl/train.py', 'replay_waypoints.py', 'record_successes.py'))
                   for line in listing.splitlines() if line.split()):
                state['status'] = 'stopped_other_simulator'
                break
            trial.update(status='running', started=time.time())
            state['active_trial'] = trial['index']
            save(out / 'status.json', state)
            print(f"Starting {trial['index']}/10 {trial['name']}", flush=True)
            stdout = out / f"{trial['index']:02d}_{trial['name']}.log"
            with stdout.open('w') as stream:
                child = subprocess.Popen(trial['command'], cwd=ROOT, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                last_progress, last_iteration = time.monotonic(), -1
                while child.poll() is None:
                    time.sleep(5)
                    content = stdout.read_text(errors='replace')
                    progress = re.findall(r'Learning iteration\s+(\d+)', content)
                    if progress and int(progress[-1]) != last_iteration:
                        last_iteration = int(progress[-1])
                        last_progress = time.monotonic()
                    trial['last_iteration'] = last_iteration
                    trial['elapsed_s'] = round(time.time() - trial['started'])
                    save(out / 'status.json', state)
                    if trial['elapsed_s'] > 1200 or time.monotonic() - last_progress > 240:
                        trial['status'] = 'timeout'
                        stop_child()
                        break
                trial['exit_code'] = child.returncode
            matches = sorted(LOGS.glob('*_' + trial['run_name']))
            if matches:
                trial['log_dir'] = str(matches[-1])
            if child.returncode == 0 and matches and (matches[-1] / 'batch_complete.json').exists():
                trial['status'] = 'completed'
                trial['completion'] = json.loads((matches[-1] / 'batch_complete.json').read_text())
                try:
                    trial['metrics'] = metrics(matches[-1])
                except Exception as error:
                    trial['metrics_error'] = str(error)
                state['completed'] += 1
            elif trial['status'] != 'timeout':
                trial['status'] = 'failed'
            save(out / 'status.json', state)
            print(f"Trial {trial['index']}: {trial['status']}, exit={child.returncode}", flush=True)
            child = None
            if trial['status'] != 'completed':
                state['status'] = 'stopped_failed_trial'
                break
        else:
            state['status'] = 'completed'
        # Compare only identical-budget trials; terrain level then reward are
        # screening signals, not a claim of target-stair benchmark performance.
        def score(trial):
            m = trial.get('metrics', {})
            return (max((v for k, v in m.items() if 'terrain_levels' in k), default=-1),
                    max((v for k, v in m.items() if 'mean_reward' in k), default=-1e30))
        ranked = sorted((t for t in plan if t['status'] == 'completed'), key=score, reverse=True)
        save(out / 'ranking.json', {'kind': 'preliminary_training_metrics', 'trials': ranked})
    except BaseException as error:
        state.update(status='interrupted', error=str(error))
        raise
    finally:
        stop_child()
        state['updated'] = time.time()
        save(out / 'status.json', state)
        lock.close()
    if state['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
