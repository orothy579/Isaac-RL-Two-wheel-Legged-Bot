"""Use Isaac Lab's standard, unmasked height scan for the staged training tasks."""
from copy import deepcopy
from dataclasses import fields

from isaaclab.envs.mdp import height_scan
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg


def use_default_height_map(cfg):
    """Keep the grid/timing, replace FOV processing and custom scan corruption.

    Do not add observations to a group that did not already consume a height map.
    In particular, the flat actor stays proprioceptive; its critic keeps the grid.
    No reward, action, terrain, or actuator settings are touched.
    """
    old = cfg.scene.height_scanner
    if old is None:
        return
    sensor = RayCasterCfg()
    for field in fields(RayCasterCfg):
        if field.name != 'class_type' and hasattr(old, field.name):
            setattr(sensor, field.name, deepcopy(getattr(old, field.name)))
    cfg.scene.height_scanner = sensor
    for name in ('none_stack_policy', 'none_stack_critic'):
        term = getattr(getattr(cfg.observations, name), 'height_scan', None)
        if term is not None:
            term.func = height_scan
            term.params = {'sensor_cfg': SceneEntityCfg('height_scanner')}
    stair = getattr(cfg.commands, 'stair_event', None)
    if stair is not None:
        stair.use_fov_mask = False
