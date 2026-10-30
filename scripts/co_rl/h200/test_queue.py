"""CPU-only checks for GPU scheduling and exact student transfer arguments."""
import json
from pathlib import Path
import tempfile
import unittest

import dispatch
import scale_train


FAKE_WORKER = '''
import argparse, json, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('job','gpu','checkpoint','iterations'):
    p.add_argument('--'+key, required=True)
a=p.parse_args()
seed=Path(a.checkpoint)
config=json.loads(seed.read_text())
out=seed.parent
(out/(a.job+'.start')).write_text(json.dumps({'t':time.monotonic(),'gpu':a.gpu}))
if config['parallel']:
    other='student' if a.job=='gain' else 'gain'
    deadline=time.monotonic()+5
    while not (out/(other+'.start')).exists():
        if time.monotonic()>deadline:
            sys.exit(9)
        time.sleep(.01)
time.sleep(.1)
(out/(a.job+'.end')).write_text(str(time.monotonic()))
sys.exit(3 if a.job=='gain' and config.get('fail_gain') else 0)
'''


class QueueTests(unittest.TestCase):
    def execute(self, count, fail_gain=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "checkpoint.json"
            seed.write_text(json.dumps({"parallel": count > 1, "fail_gain": fail_gain}))
            worker = root / "worker.py"
            worker.write_text(FAKE_WORKER)
            gpus = [{"index": i} for i in range(count)]
            plan = dispatch.plan_jobs(gpus, seed, seed)
            reports = []
            code = dispatch.execute_jobs(plan, root, lambda **data: reports.append(data), worker)
            times = {job: {"start": json.loads((root / (job + ".start")).read_text()),
                           "end": float((root / (job + ".end")).read_text())}
                     for job in ("gain", "student")}
            return code, reports[-1], times

    def test_single_gpu_student_waits_for_gain_exit(self):
        code, status, times = self.execute(1)
        self.assertEqual(code, 0)
        self.assertEqual(status["mode"], "sequential")
        self.assertGreaterEqual(times["student"]["start"]["t"], times["gain"]["end"])
        self.assertEqual(times["gain"]["start"]["gpu"], times["student"]["start"]["gpu"])

    def test_two_gpus_overlap_on_distinct_devices(self):
        code, status, times = self.execute(2)
        self.assertEqual(code, 0)
        self.assertEqual(status["mode"], "parallel")
        self.assertLess(times["student"]["start"]["t"], times["gain"]["end"])
        self.assertLess(times["gain"]["start"]["t"], times["student"]["end"])
        self.assertNotEqual(times["gain"]["start"]["gpu"], times["student"]["start"]["gpu"])

    def test_gain_failure_does_not_drop_independent_student(self):
        code, status, times = self.execute(1, fail_gain=True)
        self.assertEqual(code, 1)
        self.assertEqual(status["results"], {"gain": 3, "student": 0})
        self.assertGreaterEqual(times["student"]["start"]["t"], times["gain"]["end"])

    def test_student_history_and_all_device_settings(self):
        command = scale_train.training_command(65536, 3000, "student.pt", "student_ft",
                                                "config.json", profile="student", gpu=1)
        def argument(name):
            return command[command.index(name) + 1]
        self.assertEqual(argument("--task"), dispatch_student_task :=
                         "Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-v1-ppo")
        self.assertNotIn("Perceptive", dispatch_student_task)
        self.assertEqual(argument("--num_policy_stacks"), "49")
        self.assertEqual(argument("--num_critic_stacks"), "2")
        self.assertEqual(argument("--warmstart_std_scale"), "1.0")
        self.assertNotIn("--warmstart_widen", command)
        self.assertEqual(argument("--device"), "cuda:1")
        self.assertIn("--/renderer/multiGpu/enabled=false", argument("--kit_args"))
        self.assertEqual(scale_train.overrides(65536, 1)["agent.device"], "cuda:1")
        self.assertEqual(scale_train.overrides(65536, 1)["env.sim.device"], "cuda:1")
        self.assertEqual(scale_train.PROFILES["student"]["iterations"], 3000)

    def test_no_gpu_rejected_before_workers_start(self):
        with self.assertRaises(RuntimeError):
            dispatch.plan_jobs([], "gain.pt", "student.pt")


if __name__ == "__main__":
    unittest.main()
