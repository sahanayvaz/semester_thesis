import tensorflow as tf 
import numpy as np 
import gym 
import multiprocessing as mp
from policy import Policy 
from value import Value 
from agent import GeneratorAgentPure
from discriminator import Discriminator
from utils import Logger 
import argparse 
import signal
from gym.envs.mujoco.humanoid import HumanoidEnv
from sklearn.utils import shuffle

class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

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

	exp_info = 'expert_humanoid'
	# initialize environment
	env = HumanoidEnv()
	env.seed(0)
	
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	logger = Logger()
	killer = GracefulKiller()

	# init qpos and qvel
	init_qpos = np.load('./mocap_expert_qpos.npy')
	init_qvel = np.load('./mocap_expert_qvel.npy')
	exp_obs = np.load('./mocap_obs.npy')
	print(exp_obs.shape)

	# policy function
	policy = Policy(obs_dim=obs_dim, act_dim=act_dim, max_kl=max_kl,
					init_logvar=init_logvar, epochs=policy_epochs, 
					logger=logger)

	# value function
	value = Value(obs_dim=obs_dim, act_dim=act_dim, epochs=value_epochs, 
				  batch_size=value_batch_size, logger=logger)

	discriminator = Discriminator(obs_dim=obs_dim, act_dim=act_dim, ent_reg_weight=1e-3,
								  epochs=2, input_type='states', loss_type='pure_gail',
								  logger=logger)
	# agent
	agent = GeneratorAgentPure(env=env, policy_function=policy, value_function=value, discriminator=discriminator,
				  		   	   gamma=gamma, lam=lam, init_qpos=init_qpos, init_qvel=init_qvel,
				  		   	   logger=logger)

	print('policy lr: %f' %policy.lr)
	print('value lr %f' %value.lr)
	print('disc lr: %f' %discriminator.lr)
	# train for num_episodes
	iteration = 0
	while iteration < max_iteration:
		print('-------- iteration %d --------' %iteration)
		# collect trajectories
		obs, uns_obs, acts, tdlams, advs = agent.collect(timesteps=20000)
		
		# update policy function using ppo
		policy.update(obs, acts, advs)

		# update value function
		value.update(obs, tdlams)

		idx = np.random.randint(low=0, high=exp_obs.shape[0], size=uns_obs.shape[0])
		expert = exp_obs[idx, :]
		gen_acc, exp_acc, total_acc = discriminator.update(exp_obs=expert, gen_obs=uns_obs)
		print('gen_acc: %f, exp_acc: %f, total_acc: %f' %(gen_acc, exp_acc, total_acc))
		
		if iteration % 50 == 0:
			print('saving...')
			# save the experiment logs
			filename = './model_inter/stats_' + exp_info + '_' + str(iteration)
			logger.dump(filename)

			# save session
			filename = './model_inter/model_' + exp_info + '_' + str(iteration)
			policy.save_session(filename)

		if killer.kill_now:
			break
		# update episode number
		iteration += 1
		
	# save the experiment logs
	filename = './model/stats_' + exp_info
	logger.dump(filename)

	# save session
	filename = './model/model_' + exp_info
	policy.save_session(filename)

	# close everything
	policy.close_session()
	value.close_session()
	env.close()

if __name__ == "__main__":
	main()