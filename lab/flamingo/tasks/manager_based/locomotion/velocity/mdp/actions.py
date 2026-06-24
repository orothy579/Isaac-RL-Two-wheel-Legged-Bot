from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

# 기존 JointVelocityAction / JointVelocityActionCfg import 경로는
# 현재 프로젝트 mdp 구조에 맞춰 조정하세요.
import lab.flamingo.tasks.constraint_based.locomotion.velocity.mdp as mdp

class MaskedJointVelocityAction(mdp.JointVelocityAction):
    """Joint velocity action with env-wise action scale mask."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self._wheel_scale_mask = torch.ones(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )

    def process_actions(self, actions: torch.Tensor):
        # 기존 JointVelocityAction의 action processing 수행
        super().process_actions(actions)

        # processed action에 env-wise mask 적용
        #
        # Isaac Lab 버전에 따라 내부 변수 이름이 다를 수 있음.
        # 보통 _processed_actions 또는 _raw_actions 계열을 사용함.
        if hasattr(self, "_processed_actions"):
            self._processed_actions *= self._wheel_scale_mask
        elif hasattr(self, "processed_actions"):
            self.processed_actions *= self._wheel_scale_mask
        else:
            raise AttributeError(
                "Cannot find processed action buffer in MaskedJointVelocityAction."
            )


@configclass
class MaskedJointVelocityActionCfg(mdp.JointVelocityActionCfg):
    class_type: type[ActionTerm] = MaskedJointVelocityAction