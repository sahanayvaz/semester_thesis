import tensorflow as tf 
import numpy as np 
import gym 
import multiprocessing as mp
from policy import Policy 
from value import Value 
from agent import ExpertAgent
from discriminator import Discriminator
from utils import Logger 
import argparse 
import signal
from gym.envs.mujoco.humanoid import HumanoidEnv
from sklearn.utils import shuffle

env = HumanoidEnv()
print(env.observation_space.shape[0])
print(env.action_space.shape[0])

qpos = np.load('./mocap_expert_qpos.npy')
qvel = np.load('./mocap_expert_qvel.npy')

print(qpos.shape[1])
obs = []
for i in range(qpos.shape[1]):
	env.set_state(qpos[:, i], qvel[:, i])
	obs.append(env.get_pure_egocentric())
obs = np.array(obs)
print(obs.shape)
np.save('mocap_pure_ego', obs)