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

def main():
	max_iteration = 5000
	episodes_per_batch = 20
	max_kl = 0.01
	init_logvar = -1
	policy_epochs = 5
	value_epochs = 10
	value_batch_size = 256
	gamma = 0.995
	lam = .97

	# initialize environment
	env = HumanoidEnv()
	env.seed(0)
	
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	logger = Logger()

	# init qpos and qvel
	init_qpos = np.load('./mocap_expert_qpos.npy')
	init_qvel = np.load('./mocap_expert_qvel.npy')

	# policy function
	policy = Policy(obs_dim=obs_dim, act_dim=act_dim, max_kl=max_kl,
					init_logvar=init_logvar, epochs=policy_epochs, 
					logger=logger)

	session_to_restore = '/Users/sayvaz/Desktop/humanoid_gail_results/model_ego_inter/model_humanoid_ego_1700'
	stats_to_recover = '/Users/sayvaz/Desktop/humanoid_gail_results/model_ego_inter/stats_humanoid_ego_1700'
	scale, offset = policy.restore_session(session_to_restore=session_to_restore, stats_to_recover=stats_to_recover)
	
	# expert agent
	agent = ExpertAgent(env=env, policy_function=policy, scale=scale, offset=offset,
				  		init_qpos=init_qpos, init_qvel=init_qvel, logger=logger)

	agent.collect(episodes_per_batch=20)

	# close everything
	policy.close_session()

if __name__ == "__main__":
	main()