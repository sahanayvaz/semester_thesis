'''
This code is heavily influenced by deepmind's mocap_demo, and
using dp's amc parser to obtain qpos and qvel of mocap data
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#import time 
#import matplotlib.pyplot as plt

import copy
# Internal dependencies.
from absl import app
from absl import flags

import numpy as np

from dm_control.suite import humanoid_CMU
from dm_control.suite.utils import parse_amc
from gym.envs.mujoco.humanoid import HumanoidEnv

def flatten_obs(obs_dict):
  obs = []
  for key in obs_dict:
    #print(obs_dict[key].shape)
    obs.append(obs_dict[key].ravel())
  obs = np.concatenate([part_obs for part_obs in obs], axis=0)
  obs = np.expand_dims(obs, axis=0)
  return obs

def main(filename):
  env = humanoid_CMU.run()
  # Parse and convert specified clip.
  converted = parse_amc.convert(filename,
                                env.physics, env.control_timestep())

  max_frame = (converted.qvel.shape[1])
  trajectory = []
  reward = []
  #pos = []
  width = 480
  height = 480
  video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

  print(converted.qpos.shape)
  print(converted.qvel.shape)

  for i in range(max_frame):
    #print('we are at frame %d' %i)
    p_i = converted.qpos[:, i]
    v_i = converted.qvel[:, i]
    with env.physics.reset_context():
      env.physics.data.qpos[:] = p_i
      env.physics.data.qvel[:] = v_i    
    trajectory.append(obs['observations'])
  trajectory = np.array([traj for traj in trajectory])
  return trajectory, converted.qpos[:,0:max_frame], converted.qvel, np.sum(reward)

if __name__ == '__main__':
  basefile = '/Users/sayvaz/Desktop/mocap_utils/cmu_mocap/08_0'
  endfile = '.amc'

  trajectories = []
  positions = []
  velocities = []
  reward_stats = []
  pos_init = []
  vel_init = []
  length = 0
  for i in range(11):
    print('we are at iteration %d' %i)
    i += 1
    filename = basefile + str(i) + endfile
    traj, qpos, qvel, reward = main(filename)
    trajectories.append(traj)
    positions.append(qpos)
    velocities.append(qvel)
    reward_stats.append(reward)
    pos_init.append(qpos[:, 0])
    vel_init.append(qvel[:, 0])
  trajectories = np.concatenate(trajectories)
  positions = np.concatenate([qpos for qpos in positions], axis=1)
  velocities = np.concatenate([qvel for qvel in velocities], axis=1)
  pos_init = np.array(pos_init)
  vel_init = np.array(vel_init)
  print(trajectories.shape)
  print(positions.shape)
  print(velocities.shape)
  print(np.mean(reward_stats))
  print(pos_init.shape)
  print(vel_init.shape)

  
  #np.save('./expert_traj/mocap_expert_traj.npy', trajectories)
  np.save('./mocap_expert_qpos.npy', positions)
  np.save('./mocap_expert_qvel.npy', velocities)
  np.save('./mocap_expert_qpos_init.npy', pos_init)
  np.save('./mocap_expert_qvel_init.npy', vel_init)