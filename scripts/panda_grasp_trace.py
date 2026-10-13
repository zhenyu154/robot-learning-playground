"""Optional passive ACT/Panda grasp trace. No extra inference or action changes.

Capture the full output of normal ACT forwards through a read-only hook. JSONL
contains reset, prediction, per-control-step before/after state, and end events.
Contact forces are solver estimates, not real tactile measurements. Contact
geometry and center height alone are not a certificate of a stable grasp.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch


def cube_bottom_clearance(center, rotation, half_sizes, plane_z):
    """Lowest world-Z corner of an oriented box above a horizontal plane."""
    return float(np.asarray(center)[2] - np.abs(np.asarray(rotation).reshape(3, 3)[2]) @ np.asarray(half_sizes) - plane_z)


class PandaStateProbe:
    def __init__(self, env):
        self.base = env.unwrapped
        self.model = self.base.model
        self.data = self.base.data
        self.cube = self.model.geom('block').id
        self.floor = self.model.geom('floor').id
        self.actuator = self.model.actuator('fingers_actuator').id
        if self.model.geom_type[self.cube] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError('grasp trace expects a box-shaped cube')
        if self.model.geom_type[self.floor] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError('grasp trace expects the named floor plane')
        self.joints = [i for i in range(self.model.njnt)
                       if (self.model.joint(i).name or '').startswith(('left_', 'right_'))
                       and self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        if not self.joints:
            raise ValueError('grasp trace could not identify actual gripper hinge joints')
        self.left_geoms, self.right_geoms = set(), set()
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            while body:
                name = self.model.body(body).name or ''
                if name.startswith('left_'):
                    self.left_geoms.add(geom)
                    break
                if name.startswith('right_'):
                    self.right_geoms.add(geom)
                    break
                body = int(self.model.body_parentid[body])
        self.pad_bodies = {side: self.model.body(side + '_pad').id for side in ('left', 'right')}

    def geometry_name(self, geom):
        return self.model.geom(geom).name or f'geom_{geom}'

    def snapshot(self):
        model, data = self.model, self.data
        floor_rotation = data.geom_xmat[self.floor].reshape(3, 3)
        if not np.allclose(floor_rotation[:, 2], [0, 0, 1], atol=1e-6):
            raise ValueError('cube-bottom clearance requires a horizontal floor')
        cube_xyz = data.geom_xpos[self.cube].copy()
        rotation = data.geom_xmat[self.cube].reshape(3, 3).copy()
        contacts = []
        flags = {'cube_table_active': False, 'cube_left_active': False,
                 'cube_right_active': False, 'gripper_table_active': False}
        normal_forces = {key: 0.0 for key in flags}
        gripper_geoms = self.left_geoms | self.right_geoms
        for index in range(data.ncon):
            contact = data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            pair = {g1, g2}
            relevant = self.cube in pair or (self.floor in pair and bool(pair & gripper_geoms))
            if not relevant:
                continue
            # Inactive detected contacts have no associated solver force.
            active = int(contact.efc_address) >= 0
            force = np.zeros(6)
            if active:
                mujoco.mj_contactForce(model, data, index, force)
            contact_normal = max(0.0, float(force[0]))
            keys = []
            if pair == {self.cube, self.floor}:
                keys.append('cube_table_active')
            if self.cube in pair and pair & self.left_geoms:
                keys.append('cube_left_active')
            if self.cube in pair and pair & self.right_geoms:
                keys.append('cube_right_active')
            if self.floor in pair and pair & gripper_geoms:
                keys.append('gripper_table_active')
            for key in keys:
                flags[key] = flags[key] or active
                normal_forces[key] += contact_normal
            contacts.append({
                'geom1': self.geometry_name(g1), 'geom2': self.geometry_name(g2),
                'distance_m': float(contact.dist), 'constraint_active': active,
                'force_torque_contact_frame': force.tolist(),
                'position_world': contact.pos.copy().tolist(),
            })
        finger_joints = {}
        for joint in self.joints:
            q, v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
            finger_joints[model.joint(joint).name] = {
                'position_rad': float(data.qpos[q]), 'velocity_rad_s': float(data.qvel[v]),
            }
        pads = {side: data.xpos[body].copy() for side, body in self.pad_bodies.items()}
        # These are pad BODY origins, not the actual contact points/separation.
        result = {
            'simulation_time_s': float(data.time),
            'cube_xyz': cube_xyz.tolist(), 'cube_rotation_world': rotation.tolist(),
            'cube_bottom_clearance_m': cube_bottom_clearance(
                cube_xyz, rotation, model.geom_size[self.cube], data.geom_xpos[self.floor][2]),
            'tcp_xyz': data.sensor('2f85/pinch_pos').data.copy().tolist(),
            'tcp_velocity_xyz_m_s': data.sensor('2f85/pinch_vel').data.copy().tolist(),
            'tcp_target_xyz': data.mocap_pos[0].copy().tolist(),
            'gripper_control_target_0_to_255': float(data.ctrl[self.actuator]),
            'finger_joints': finger_joints,
            'pad_body_origins_xyz': {side: p.tolist() for side, p in pads.items()},
            'pad_body_origin_distance_m': float(np.linalg.norm(pads['left'] - pads['right'])),
            'contact_flags': flags, 'summed_contact_normal_forces_n': normal_forces,
            'contacts': contacts,
        }
        return result


class GraspTrace:
    """Read-only hook on ACT's ordinary model forward; JSONL writes are optional."""
    def __init__(self, path: Path, policy, postprocessor, env, metadata: dict):
        if getattr(policy.config, 'temporal_ensemble_coeff', None) is not None:
            raise ValueError('this trace expects the standard action queue, not temporal ensembling')
        if getattr(policy.config, 'n_action_steps', 0) <= 0:
            raise ValueError('invalid action queue length for tracing')
        self.probe = PandaStateProbe(env)
        self.policy = policy
        self.postprocessor = postprocessor
        self.pending = None
        self.chunk_id = 0
        self.offset = 0
        self.episode = None
        self.step = None
        self.before = None
        self.handle = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('x', buffering=1)
        try:
            self.write({'event': 'metadata', 'schema_version': 1, **metadata,
                        'chunk_size': policy.config.chunk_size,
                        'n_action_steps': policy.config.n_action_steps,
                        'units': {'actions': 'unnormalized policy action units, before environment scaling/clipping',
                                  'positions': 'meters', 'hinge_positions': 'radians',
                                  'contact_forces': 'MuJoCo solver estimate in contact frame'},
                        'notes': ['no extra ACT predictions; full chunks captured from original forwards',
                                  'gripper target is not finger position or grip force',
                                  'contact flags are constraint-active contacts, not binary grasp success',
                                  'cube/contact measurements are diagnostic only, never policy inputs',
                                  'the traced run has logging overhead; simulation stepping is unchanged']})
            self.handle = policy.model.register_forward_hook(self.capture)
        except BaseException:
            self.file.close()
            raise

    def write(self, row):
        self.file.write(json.dumps(row, allow_nan=False) + '\n')

    def capture(self, module, inputs, output):
        if self.pending is not None:
            raise RuntimeError('more than one model prediction in a single traced control step')
        if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
            raise RuntimeError('ACT forward output changed; cannot trace safely')
        chunk = output[0]
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[-1] != 4:
            raise RuntimeError('expected an ACT chunk with shape [1, time, 4]')
        self.pending = chunk.detach().clone()
        # Return None, leaving the original model outputs untouched.

    def reset(self, episode: int, scheduled_position):
        self.pending = None
        self.chunk_id = 0
        self.offset = 0
        self.episode = episode
        self.write({'event': 'reset', 'episode_index': episode,
                    'scheduled_position': scheduled_position, 'state': self.probe.snapshot()})

    def before_step(self, step: int, observation_state=None):
        self.step = step
        self.before = self.probe.snapshot()
        self.observation_state = (observation_state.detach().cpu().float().reshape(-1).tolist()
                                  if observation_state is not None else None)

    def after_prediction(self, policy_action):
        if self.pending is not None:
            normalized = self.pending
            self.pending = None
            self.chunk_id += 1
            self.offset = 0
            # The saved postprocessor is an unnormalizer + device step. A clone
            # ensures processing a full diagnostic chunk never alters the queue.
            with torch.inference_mode():
                actions = self.postprocessor.process_action(normalized.reshape(-1, 4).clone())
            actions = actions.detach().cpu().float().reshape(-1, 4)
            self.chunk_actions = actions
            self.write({'event': 'prediction', 'episode_index': self.episode,
                        'step': self.step, 'chunk_id': self.chunk_id,
                        'queue_executed_prefix_length': self.policy.config.n_action_steps,
                        'actions': actions.tolist(), 'state_before_prediction': self.before,
                        'raw_observation_robot_state': self.observation_state})
        if not self.chunk_id or self.offset >= self.policy.config.n_action_steps:
            raise RuntimeError('trace queue accounting differs from ACT execution')
        actual = policy_action.detach().cpu().float().reshape(-1)
        if not torch.allclose(actual, self.chunk_actions[self.offset], rtol=1e-5, atol=1e-6):
            raise RuntimeError('captured chunk action differs from actual unnormalized policy action')

    def after_step(self, policy_action, requested_action, reward, terminated, truncated):
        self.write({'event': 'step', 'episode_index': self.episode, 'step': self.step,
                    'chunk_id': self.chunk_id, 'chunk_offset': self.offset,
                    'policy_action': policy_action.detach().cpu().float().reshape(-1).tolist(),
                    'requested_action': requested_action.detach().cpu().float().reshape(-1).tolist(),
                    'reward': float(reward), 'terminated': bool(terminated), 'truncated': bool(truncated),
                    'before': self.before, 'after': self.probe.snapshot()})
        self.offset += 1

    def end_episode(self, success, steps, reward):
        self.write({'event': 'end', 'episode_index': self.episode,
                    'success': bool(success), 'steps': int(steps), 'reward': float(reward)})

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.file.close()
