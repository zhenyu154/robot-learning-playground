"""Passive trace correctness: tiny MuJoCo scene, real ACT queue, no GUI/XPU."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch
from torch import nn
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

from scripts.panda_grasp_trace import GraspTrace, PandaStateProbe, cube_bottom_clearance
from scripts.summarize_grasp_trace import summarize

XML = '''<mujoco><option gravity="0 0 -9.81"/>
<worldbody>
  <geom name="floor" type="plane" size="1 1 0.01"/>
  <body name="block" pos="0 0 0.0199"><freejoint/>
    <geom name="block" type="box" size="0.02 0.02 0.02" mass="0.1"/>
  </body>
  <body name="target" pos="0.5 0 0.1" mocap="true"><site name="pinch"/></body>
  <body name="left_pad" pos="0.04 0 0.03">
    <joint name="left_driver_joint" type="hinge" axis="0 0 1"/>
    <geom name="left_pad1" type="box" size="0.01 0.01 0.01" mass="0.1"/>
  </body>
  <body name="right_pad" pos="-0.04 0 0.03">
    <joint name="right_driver_joint" type="hinge" axis="0 0 1"/>
    <geom name="right_pad1" type="box" size="0.01 0.01 0.01" mass="0.1"/>
  </body>
</worldbody>
<actuator><motor name="fingers_actuator" joint="left_driver_joint"/></actuator>
<sensor><framepos name="2f85/pinch_pos" objtype="site" objname="pinch"/>
<framelinvel name="2f85/pinch_vel" objtype="site" objname="pinch"/></sensor>
</mujoco>'''


def env():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = SimpleNamespace(model=model, data=data)
    return SimpleNamespace(unwrapped=base)


class PredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        values = torch.zeros(1, 50, 4)
        values[:, :, 3] = 0.5
        values[:, 8:, 2] = 1.0
        self.values = nn.Parameter(values)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return self.values.clone(), (None, None)


def policy():
    p = ACTPolicy.__new__(ACTPolicy)
    nn.Module.__init__(p)
    p.config = ACTConfig(
        input_features={'observation.state': PolicyFeature(FeatureType.STATE, (18,))},
        output_features={'action': PolicyFeature(FeatureType.ACTION, (4,))},
        device='cpu', push_to_hub=False, chunk_size=50, n_action_steps=5,
        use_vae=False, pretrained_backbone_weights=None,
    )
    p.model = PredictionModel()
    p.reset()
    return p


class ProbeTests(unittest.TestCase):
    def test_clearance_accounts_for_tilt(self):
        c = [0, 0, 0.02]
        self.assertAlmostEqual(cube_bottom_clearance(c, np.eye(3), [0.02]*3, 0), 0)
        angle = np.pi / 4
        rotation = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        self.assertLess(cube_bottom_clearance(c, rotation, [0.02]*3, 0), 0)

    def test_snapshot_is_read_only_and_gripper_target_is_separate(self):
        e = env()
        e.unwrapped.data.ctrl[0] = 255
        mujoco.mj_forward(e.unwrapped.model, e.unwrapped.data)
        before = (e.unwrapped.data.qpos.copy(), e.unwrapped.data.qvel.copy(),
                  e.unwrapped.data.ctrl.copy(), e.unwrapped.data.mocap_pos.copy(), e.unwrapped.data.time)
        state = PandaStateProbe(e).snapshot()
        self.assertEqual(state['gripper_control_target_0_to_255'], 255)
        self.assertEqual(state['finger_joints']['left_driver_joint']['position_rad'], 0)
        self.assertTrue(state['contact_flags']['cube_table_active'])
        self.assertGreater(state['summed_contact_normal_forces_n']['cube_table_active'], 0)
        for a, b in zip(before[:-1], (e.unwrapped.data.qpos, e.unwrapped.data.qvel, e.unwrapped.data.ctrl, e.unwrapped.data.mocap_pos)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(before[-1], e.unwrapped.data.time)
        json.dumps(state, allow_nan=False)


class TraceTests(unittest.TestCase):
    def test_real_act_queue_and_unnormalizer_unchanged_no_extra_forward(self):
        original = policy()
        traced = copy.deepcopy(original)
        _, post = make_act_pre_post_processors(original.config, {
            'action': {'mean': torch.tensor([0,0,0,1.0]), 'std': torch.tensor([1,1,0.5,1.0])},
        })
        expected = []
        obs = {'observation.state': torch.zeros(1, 18)}
        for _ in range(10):
            expected.append(post.process_action(original.select_action(obs)).clone())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            trace = GraspTrace(path, traced, post, env(), {'checkpoint': 'unit-test', 'max_steps': 150})
            trace.reset(0, {'x': 0.44, 'y': -0.05625})
            for step in range(1, 11):
                trace.before_step(step)
                action = post.process_action(traced.select_action(obs))
                trace.after_prediction(action)
                torch.testing.assert_close(action, expected[step-1], rtol=0, atol=0)
                trace.after_step(action, action, 0, False, step==10)
            trace.end_episode(False, 10, 0)
            trace.close()
            self.assertEqual(traced.model.calls, original.model.calls)
            self.assertEqual(traced.model.calls, 2)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            chunks = [r for r in events if r['event']=='prediction']
            self.assertEqual(len(chunks), 2)
            self.assertEqual(len(chunks[0]['actions']), 50)
            self.assertAlmostEqual(chunks[0]['actions'][8][2], 0.5)
            steps = [r for r in events if r['event']=='step']
            self.assertEqual([r['chunk_offset'] for r in steps], list(range(5))*2)
            self.assertEqual(len(traced.model._forward_hooks), 0)
            with self.assertRaises(FileExistsError):
                GraspTrace(path, traced, post, env(), {})

    def test_trace_detects_queue_mismatch(self):
        p = policy()
        post = SimpleNamespace(process_action=lambda a:a.clone())
        with tempfile.TemporaryDirectory() as directory:
            trace = GraspTrace(Path(directory)/'trace.jsonl', p, post, env(), {})
            try:
                trace.reset(0, None)
                trace.before_step(1)
                action = p.select_action({'observation.state': torch.zeros(1,18)})
                with self.assertRaisesRegex(RuntimeError,'differs from actual'):
                    trace.after_prediction(action + 1)
            finally:
                trace.close()
            self.assertEqual(len(p.model._forward_hooks), 0)

    def test_summary_distinguishes_commands_clearance_and_late_chunk(self):
        state = PandaStateProbe(env()).snapshot()
        rows = [
            {'event':'metadata', 'checkpoint':'test', 'max_steps':150, 'n_action_steps':5},
            {'event':'step','episode_index':0,'step':1,'requested_action':[0,0,0,2], 'after':state},
            {'event':'prediction','episode_index':0,'step':2,'chunk_id':1,
             'queue_executed_prefix_length':5,'actions':[[0,0,0,1]]*8+[[0,0,0.5,1]]*42},
            {'event':'end','episode_index':0,'success':False,'steps':1,'reward':0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'trace.jsonl'
            p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            s = summarize(p)
            r = s['episodes'][0]
            self.assertEqual(r['first_close_command_step'], 1)
            self.assertIsNone(r['first_post_close_clearance_without_table_contact_step'])
            self.assertIsNone(r['first_post_close_positive_z_command_step'])
            self.assertEqual(r['chunks_with_positive_z_only_beyond_executed_prefix'][0]['first_positive_z_offset_zero_based'], 8)


if __name__ == '__main__':
    unittest.main()
