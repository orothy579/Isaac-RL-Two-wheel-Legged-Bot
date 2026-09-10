import math
import unittest
from types import SimpleNamespace
import torch
from scripts.co_rl.staged_retrain.approach_proposal import ScanApproach, estimate_approach, timing_eligible


class ApproachTests(unittest.TestCase):
    def scene(self, angle=0.0):
        x = torch.linspace(0, 0.8, 161)
        y = torch.linspace(-0.3, 0.3, 13)
        edge = 0.3 - math.tan(angle) * y[:, None]
        z = (x[None, :] >= edge).float()[None] * 0.15
        return x, y, z, torch.ones_like(z, dtype=torch.bool), torch.zeros(1)

    def test_distance_and_signed_angle(self):
        for degrees in (0, -15, 15, 35):
            a = math.radians(degrees)
            result = estimate_approach(*self.scene(a))
            self.assertTrue(result['valid'].item())
            self.assertAlmostEqual(result['angle_rad'].item(), a, delta=0.012)
            self.assertAlmostEqual(result['distance_m'].item(), 0.3 * math.cos(a), delta=0.006)

    def test_missing_and_nonfinite_cells(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            x, y, z, mask, ground = self.scene()
            z.fill_(value)
            result = estimate_approach(x, y, z, mask, ground)
            self.assertFalse(result['valid'].item())
            self.assertTrue(torch.isfinite(result['distance_m']).all())
        x, y, z, mask, ground = self.scene()
        mask.zero_()
        self.assertFalse(estimate_approach(x, y, z, mask, ground)['valid'].item())

    def test_floor_ramp_and_unseen_lower_floor(self):
        x, y, z, mask, ground = self.scene()
        for surface in (torch.zeros_like(z), torch.ones_like(z)*0.15,
                        x[None,None,:].expand_as(z)*0.2):
            self.assertFalse(estimate_approach(x,y,surface,mask,ground)['valid'].item())
        mask[..., :61] = False
        self.assertFalse(estimate_approach(x,y,z,mask,ground)['valid'].item())

    def test_trigger_requires_forward_aligned_approach(self):
        result = estimate_approach(*self.scene())
        speed = torch.tensor([[0.7, 0.]])
        for cmd, allowed in ((0.7, True), (-0.7, False), (0., False)):
            self.assertEqual(timing_eligible(result,speed,torch.tensor([cmd]))[0].item(),allowed)
        self.assertFalse(timing_eligible(result,-speed,torch.tensor([0.7]))[0].item())
        result = estimate_approach(*self.scene(math.radians(35)))
        self.assertFalse(timing_eligible(result,speed,torch.tensor([0.7]))[0].item())

    def test_lateral_velocity_changes_contact_time(self):
        result = estimate_approach(*self.scene(math.radians(15)))
        _, t1 = timing_eligible(result,torch.tensor([[0.7,0.0]]),torch.tensor([0.7]))
        _, t2 = timing_eligible(result,torch.tensor([[0.7,0.2]]),torch.tensor([0.7]))
        self.assertLess(t2.item(),t1.item())

    def test_scanner_adapter_respects_order_and_visibility(self):
        for ordering in ('xy', 'yx'):
            sensor_cfg = SimpleNamespace(ray_alignment='yaw',
                offset=SimpleNamespace(pos=(0.,0.,20.),rot=(1.,0.,0.,0.)),
                pattern_cfg=SimpleNamespace(size=(1.1,1.1),resolution=.1,ordering=ordering))
            adapter = ScanApproach(sensor_cfg,'cpu',use_fov_mask=True)
            z = (adapter.x[None,:] > .3).float().expand(len(adapter.y),-1)*.15
            if ordering=='yx':
                z=z.T
            hits=torch.zeros(1,z.numel(),3);hits[...,2]=z.flatten()
            data=SimpleNamespace(ray_hits_w=hits,valid_mask=torch.ones(1,z.numel()))
            result=adapter.estimate(data,torch.zeros(1,1))
            self.assertTrue(result['valid'].item())
            self.assertAlmostEqual(result['distance_m'].item(),.3,delta=.051)
            data.valid_mask.zero_()
            self.assertFalse(adapter.estimate(data,torch.zeros(1,1))['valid'].item())
            data.valid_mask.fill_(1)
            self.assertFalse(adapter.estimate(data,torch.full((1,1),float('inf')))['valid'].item())
            data.valid_mask=None
            with self.assertRaisesRegex(ValueError,'mask required'):
                adapter.estimate(data,torch.zeros(1,1))
            # The requested default Isaac grid has no FOV mask at all.
            unmasked = ScanApproach(sensor_cfg,'cpu')
            self.assertTrue(unmasked.estimate(data,torch.zeros(1,1))['valid'].item())

    def test_timing_rejects_outside_distance_or_contact_window(self):
        for d, speed in ((.1,.7),(.5,.7),(.3,.11),(.3,3.0)):
            result={'distance_m':torch.tensor([d]),'angle_rad':torch.zeros(1),'valid':torch.tensor([True])}
            allowed,_=timing_eligible(result,torch.tensor([[speed,0.]]),torch.tensor([.7]))
            self.assertFalse(allowed.item())

    def test_plane_termination_is_a_per_environment_tensor(self):
        # Execute the actual function without importing Isaac Sim. This reproduces
        # the manager's tensor contract that originally stopped flat training.
        import ast
        from pathlib import Path
        tree=ast.parse(Path('lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/terminations.py').read_text())
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='terrain_out_of_bounds')
        for argument in func.args.args:
            argument.annotation=None
        func.returns=None
        namespace={'torch':torch,'SceneEntityCfg':lambda name:None}
        exec(compile(ast.Module(body=[func],type_ignores=[]),'<termination>','exec'),namespace)
        env=SimpleNamespace(num_envs=4,device='cpu',scene=SimpleNamespace(cfg=SimpleNamespace(
            terrain=SimpleNamespace(terrain_type='plane'))))
        result=namespace['terrain_out_of_bounds'](env)
        self.assertEqual(result.shape,(4,))
        self.assertEqual(result.dtype,torch.bool)
        self.assertEqual(len(result.nonzero(as_tuple=True)[0]),0)


if __name__ == '__main__':
    unittest.main()
