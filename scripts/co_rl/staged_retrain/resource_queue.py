"""Wait for resources, then launch the authorized two-trial comparison once.

Never stop another user's job. An OOM/resource failure is recorded, not retried
indefinitely. This lightweight process runs outside the simulator's memory cap.
"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/retrain_20260909'
COMPARISON = OUT / 'controlled_flat_defaultmap_queued_20260910'
STATUS = OUT / 'resource_queue_status.json'
LABEL = 'flat-default-map-queued-20260910'
UNIT = 'flamingo-' + LABEL + '.service'


def save(state):
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATUS)


def resources():
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:')) / (1024 * 1024)
    gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    free_mib, utilization = [float(x.strip()) for x in gpu.split(',')]
    return {'available_ram_gib': round(available, 2), 'free_gpu_mib': free_mib,
            'gpu_utilization_percent': utilization}


def main():
    if COMPARISON.exists():
        raise FileExistsError(f'Refusing duplicate comparison: {COMPARISON}')
    state = {'status': 'waiting_resources', 'comparison': str(COMPARISON), 'unit': UNIT,
             'requirements': {'available_ram_gib': 8, 'free_gpu_mib': 6500,
                              'gpu_utilization_percent_below': 20, 'consecutive_checks': 3},
             'note': 'Two scratch models, 1000 updates each, default unmasked height map; no rough training.'}
    start, ready = time.monotonic(), 0
    while time.monotonic() - start < 24 * 3600:
        try:
            r = resources()
            state['resources'] = r
            good = r['available_ram_gib'] >= 8 and r['free_gpu_mib'] >= 6500 and r['gpu_utilization_percent'] < 20
            ready = ready + 1 if good else 0
            state['consecutive_ready_checks'] = ready
        except (subprocess.SubprocessError, ValueError) as error:
            state['resource_error'] = str(error)
            ready = 0
        state['updated_at'] = time.time()
        save(state)
        if ready >= 3:
            break
        time.sleep(30)
    else:
        state['status'] = 'expired_waiting_resources'
        save(state)
        return
    environment = dict(os.environ, FLAMINGO_JOB_SECONDS='28800')
    subprocess.run(['bash', 'scripts/co_rl/staged_retrain/launch.sh', LABEL,
                    'python', '-u', 'scripts/co_rl/staged_retrain/compare_flat.py',
                    '--output', str(COMPARISON)], cwd=ROOT, env=environment, check=True)
    state['status'] = 'worker_started'
    save(state)
    low_memory = 0
    while True:
        time.sleep(30)
        trial_status = COMPARISON / 'status.json'
        if trial_status.exists():
            status = json.loads(trial_status.read_text())
            if status['status'] in ('completed', 'failed'):
                state['status'] = status['status']
                save(state)
                return
        worker = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', UNIT]).returncode == 0
        if not worker:
            state['status'] = 'worker_stopped_before_completion'
            save(state)
            return
        try:
            state['resources'] = resources()
            low_memory = low_memory + 1 if state['resources']['available_ram_gib'] < 1.5 else 0
            if low_memory >= 2:
                subprocess.run(['systemctl', '--user', 'stop', UNIT], check=True)
                state['status'] = 'stopped_own_worker_low_host_memory'
                save(state)
                return
        except (subprocess.SubprocessError, ValueError) as error:
            state['resource_error'] = str(error)
        state['updated_at'] = time.time()
        save(state)


if __name__ == '__main__':
    main()
