"""Lightweight preflight; acquire before importing/starting Isaac Sim."""

import fcntl
from pathlib import Path


def available_memory_kib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
    raise RuntimeError('Cannot determine available RAM; refusing GPU launch')


def acquire_replay_lock(path, minimum_available_kib=6*1024*1024):
    """Keep returned file alive until exit. Never unlink a live lock's inode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Another waypoint replay is still running, including shutdown. Wait for it to exit.') from None
    if available_memory_kib() < minimum_available_kib:
        handle.close()
        raise RuntimeError('Less than 6 GiB RAM available; refusing Isaac Sim startup')
    return handle
