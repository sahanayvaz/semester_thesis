import numpy as np 
from utils import Scaler
import scipy.signal
import pickle

class GeneratorAgentPure(object):
	def __init__(self, env, policy_function, value_function, discriminator,
				gamma, lam, init_qpos, init_qvel, logger=None):
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		self.policy = policy_function
		self.value = value_function
		self.discriminator = discriminator
		self.gamma = gamma
		self.lam = lam

		self.init_qpos = init_qpos
		self.init_qvel = init_qvel

		self.scaler = Scaler(self.obs_dim)	

		# logger
		self.logger = logger
		
		# set scaler's scale and offset by collecting 5 episodes
		self.collect(timesteps=2048)

	def discount(self, x, gamma):
		return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

	def get_random(self):
		idx = np.random.randint(low=0, high=self.init_qpos.shape[1], size=1)
		return np.squeeze(self.init_qpos[:, idx]), np.squeeze(self.init_qvel[:, idx])

	def collect(self, timesteps):
		trajectories = []
		trew_stat = []

		scale, offset = self.scaler.get()

		self.logger.log('scale_offset', [scale, offset])

		buffer_time = 0
		while buffer_time < timesteps:
			unscaled_obs, scaled_obs, actions, rewards = [], [], [], []
			egocentric = []
			done = False
			obs = self.env.reset()
			qpos, qvel = self.get_random()
			# we are setting initial qpos and qvel from expert
			self.env.set_state(qpos, qvel)
			timestep = 0
			while not done and timestep < 1000:
				obs = obs.astype(np.float32).reshape(1, -1)
				unscaled_obs.append(obs)
				obs = (obs - offset) * scale 
				scaled_obs.append(obs)
				acts = self.policy.sample(obs)
				actions.append(acts.astype(np.float32).reshape(1, -1))
				obs, rew, done, _ = self.env.step(acts)
				rewards.append(rew)
				timestep += 1
				buffer_time += 1
				
			# statistics
			trew_stat.append(np.sum(rewards))

			# episode info
			traj_obs = np.concatenate(scaled_obs)
			traj_unscaled_obs = np.concatenate(unscaled_obs)
			traj_acts = np.concatenate(actions)
			#traj_rews = np.array(rewards, dtype=np.float64)
			traj_rews = np.squeeze(self.discriminator.get_rewards(traj_unscaled_obs, traj_acts))

			# scale rewards using running std of the experiment
			# traj_scaled_rews = traj_rews * np.squeeze(rew_scale)
			traj_scaled_rews = traj_rews

			# calculate discount sum of rewards
			traj_disc_rews = self.discount(traj_scaled_rews, self.gamma)

			# calculate advantages
			traj_values = self.value.predict(traj_obs)

			deltas = traj_scaled_rews - traj_values + np.append(traj_values[1:] * self.gamma, 0)
			traj_advantages = self.discount(deltas, self.gamma*self.lam)

			trajectory = {'observations' : traj_obs,
						  'actions': traj_acts,
						  'tdlam': traj_disc_rews,
						  'advantages': traj_advantages,
						  'unscaled_obs': traj_unscaled_obs}
			trajectories.append(trajectory)

		# update observation scaler
		uns_obs = np.concatenate([t['unscaled_obs'] for t in trajectories])
		self.scaler.update(uns_obs)

		# update rewards scaler
		#uns_rews = np.concatenate([t['unscaled_rews'] for t in trajectories])
		#self.rew_scaler.update(uns_rews)
		observations = np.concatenate([t['observations'] for t in trajectories])
		actions = np.concatenate([t['actions'] for t in trajectories])
		tdlam = np.concatenate([t['tdlam'] for t in trajectories])
		advantages = np.concatenate([t['advantages'] for t in trajectories])
		advantages = (advantages - np.mean(advantages)) / np.std(advantages)

		# check stats
		print('mean_trew: %f' %np.mean(trew_stat))
		self.logger.log('trew_stat', np.mean(trew_stat))
		
		return observations, uns_obs, actions, tdlam, advantages

class GeneratorAgentEgo(object):
	def __init__(self, env, policy_function, value_function, discriminator,
				gamma, lam, init_qpos, init_qvel, logger=None):
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		self.policy = policy_function
		self.value = value_function
		self.discriminator = discriminator
		self.gamma = gamma
		self.lam = lam

		self.init_qpos = init_qpos
		self.init_qvel = init_qvel

		self.scaler = Scaler(self.obs_dim)	

		# logger
		self.logger = logger
		
		# set scaler's scale and offset by collecting 5 episodes
		self.collect(timesteps=2048)

	def discount(self, x, gamma):
		return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

	def get_random(self):
		idx = np.random.randint(low=0, high=self.init_qpos.shape[1], size=1)
		return np.squeeze(self.init_qpos[:, idx]), np.squeeze(self.init_qvel[:, idx])

	def collect(self, timesteps):
		trajectories = []
		trew_stat = []

		scale, offset = self.scaler.get()

		self.logger.log('scale_offset', [scale, offset])

		buffer_time = 0
		while buffer_time < timesteps:
			unscaled_obs, scaled_obs, actions, rewards = [], [], [], []
			egocentric = []
			done = False
			obs = self.env.reset()
			qpos, qvel = self.get_random()
			# we are setting initial qpos and qvel from expert
			self.env.set_state(qpos, qvel)
			timestep = 0
			while not done and timestep < 1000:
				obs = obs.astype(np.float32).reshape(1, -1)
				unscaled_obs.append(obs)
				obs = (obs - offset) * scale 
				scaled_obs.append(obs)
				acts = self.policy.sample(obs)
				actions.append(acts.astype(np.float32).reshape(1, -1))
				obs, rew, done, info = self.env.step(acts)
				egocentric_feats = info['egocentric_feats']
				egocentric.append(egocentric_feats.astype(np.float32).reshape(1, -1))
				rewards.append(rew)
				timestep += 1
				buffer_time += 1
				
			# statistics
			trew_stat.append(np.sum(rewards))

			# episode info
			traj_obs = np.concatenate(scaled_obs)
			traj_unscaled_obs = np.concatenate(unscaled_obs)
			traj_acts = np.concatenate(actions)
			traj_egocentric = np.concatenate(egocentric)
			#traj_rews = np.array(rewards, dtype=np.float64)
			traj_rews = np.squeeze(self.discriminator.get_rewards(traj_egocentric, traj_acts))

			# scale rewards using running std of the experiment
			# traj_scaled_rews = traj_rews * np.squeeze(rew_scale)
			traj_scaled_rews = traj_rews

			# calculate discount sum of rewards
			traj_disc_rews = self.discount(traj_scaled_rews, self.gamma)

			# calculate advantages
			traj_values = self.value.predict(traj_obs)

			deltas = traj_scaled_rews - traj_values + np.append(traj_values[1:] * self.gamma, 0)
			traj_advantages = self.discount(deltas, self.gamma*self.lam)

			trajectory = {'observations' : traj_obs,
						  'actions': traj_acts,
						  'tdlam': traj_disc_rews,
						  'advantages': traj_advantages,
						  'unscaled_obs': traj_unscaled_obs,
						  'egocentric': traj_egocentric}
			trajectories.append(trajectory)

		# update observation scaler
		uns_obs = np.concatenate([t['unscaled_obs'] for t in trajectories])
		self.scaler.update(uns_obs)

		# update rewards scaler
		#uns_rews = np.concatenate([t['unscaled_rews'] for t in trajectories])
		#self.rew_scaler.update(uns_rews)
		egocentric = np.concatenate([t['egocentric'] for t in trajectories])
		observations = np.concatenate([t['observations'] for t in trajectories])
		actions = np.concatenate([t['actions'] for t in trajectories])
		tdlam = np.concatenate([t['tdlam'] for t in trajectories])
		advantages = np.concatenate([t['advantages'] for t in trajectories])
		advantages = (advantages - np.mean(advantages)) / np.std(advantages)

		# check stats
		print('mean_trew: %f' %np.mean(trew_stat))
		self.logger.log('trew_stat', np.mean(trew_stat))
		
		return observations, egocentric, actions, tdlam, advantages

class GeneratorAgentEgoPure(object):
	def __init__(self, env, policy_function, value_function, discriminator,
				gamma, lam, init_qpos, init_qvel, logger=None):
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		self.policy = policy_function
		self.value = value_function
		self.discriminator = discriminator
		self.gamma = gamma
		self.lam = lam

		self.init_qpos = init_qpos
		self.init_qvel = init_qvel

		self.scaler = Scaler(self.obs_dim)	

		# logger
		self.logger = logger
		
		# set scaler's scale and offset by collecting 5 episodes
		self.collect(timesteps=2048)

	def discount(self, x, gamma):
		return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

	def get_random(self):
		idx = np.random.randint(low=0, high=self.init_qpos.shape[1], size=1)
		return np.squeeze(self.init_qpos[:, idx]), np.squeeze(self.init_qvel[:, idx])

	def collect(self, timesteps):
		trajectories = []
		trew_stat = []

		scale, offset = self.scaler.get()

		self.logger.log('scale_offset', [scale, offset])

		buffer_time = 0
		while buffer_time < timesteps:
			unscaled_obs, scaled_obs, actions, rewards = [], [], [], []
			egocentric = []
			done = False
			obs = self.env.reset()
			qpos, qvel = self.get_random()
			# we are setting initial qpos and qvel from expert
			self.env.set_state(qpos, qvel)
			timestep = 0
			while not done and timestep < 1000:
				obs = obs.astype(np.float32).reshape(1, -1)
				unscaled_obs.append(obs)
				obs = (obs - offset) * scale 
				scaled_obs.append(obs)
				acts = self.policy.sample(obs)
				actions.append(acts.astype(np.float32).reshape(1, -1))
				obs, rew, done, info = self.env.step(acts)
				egocentric_feats = info['pure_egocentric_feats']
				egocentric.append(egocentric_feats.astype(np.float32).reshape(1, -1))
				rewards.append(rew)
				timestep += 1
				buffer_time += 1
				
			# statistics
			trew_stat.append(np.sum(rewards))

			# episode info
			traj_obs = np.concatenate(scaled_obs)
			traj_unscaled_obs = np.concatenate(unscaled_obs)
			traj_acts = np.concatenate(actions)
			traj_egocentric = np.concatenate(egocentric)
			#traj_rews = np.array(rewards, dtype=np.float64)
			traj_rews = np.squeeze(self.discriminator.get_rewards(traj_egocentric, traj_acts))

			# scale rewards using running std of the experiment
			# traj_scaled_rews = traj_rews * np.squeeze(rew_scale)
			traj_scaled_rews = traj_rews

			# calculate discount sum of rewards
			traj_disc_rews = self.discount(traj_scaled_rews, self.gamma)

			# calculate advantages
			traj_values = self.value.predict(traj_obs)

			deltas = traj_scaled_rews - traj_values + np.append(traj_values[1:] * self.gamma, 0)
			traj_advantages = self.discount(deltas, self.gamma*self.lam)

			trajectory = {'observations' : traj_obs,
						  'actions': traj_acts,
						  'tdlam': traj_disc_rews,
						  'advantages': traj_advantages,
						  'unscaled_obs': traj_unscaled_obs,
						  'egocentric': traj_egocentric}
			trajectories.append(trajectory)

		# update observation scaler
		uns_obs = np.concatenate([t['unscaled_obs'] for t in trajectories])
		self.scaler.update(uns_obs)

		# update rewards scaler
		#uns_rews = np.concatenate([t['unscaled_rews'] for t in trajectories])
		#self.rew_scaler.update(uns_rews)
		egocentric = np.concatenate([t['egocentric'] for t in trajectories])
		observations = np.concatenate([t['observations'] for t in trajectories])
		actions = np.concatenate([t['actions'] for t in trajectories])
		tdlam = np.concatenate([t['tdlam'] for t in trajectories])
		advantages = np.concatenate([t['advantages'] for t in trajectories])
		advantages = (advantages - np.mean(advantages)) / np.std(advantages)

		# check stats
		print('mean_trew: %f' %np.mean(trew_stat))
		self.logger.log('trew_stat', np.mean(trew_stat))
		
		return observations, egocentric, actions, tdlam, advantages

class ExpertAgent(object):
	def __init__(self, env, policy_function, scale, offset, init_qpos, init_qvel,
				 logger=None):
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		self.policy = policy_function

		self.scale = scale
		self.offset = offset

		self.init_qpos = init_qpos
		self.init_qvel = init_qvel

		# logger
		self.logger = logger
	
	def discount(self, x, gamma):
		return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

	def get_random(self):
		idx = np.random.randint(low=0, high=self.init_qpos.shape[1], size=1)
		return np.squeeze(self.init_qpos[:, idx]), np.squeeze(self.init_qvel[:, idx])

	def collect(self, episodes_per_batch):
		trajectories = []
		rew_stat = []

		for _ in range(episodes_per_batch):
			unscaled_obs, scaled_obs, actions, rewards = [], [], [], []
			done = False
			obs = self.env.reset()
			qpos, qvel = self.get_random()
			# we are setting initial qpos and qvel from expert
			self.env.set_state(qpos, qvel)
			timestep = 0
			while not done and timestep < 1000:
				self.env.render()
				obs = obs.astype(np.float32).reshape(1, -1)
				unscaled_obs.append(obs)
				obs = (obs - self.offset) * self.scale 
				scaled_obs.append(obs)
				acts = self.policy.sample(obs)
				actions.append(acts.astype(np.float32).reshape(1, -1))
				obs, rew, done, _ = self.env.step(acts)
				rewards.append(rew)
				timestep += 1
				
			# statistics
			rew_stat.append(np.sum(rewards))
		# check stats
		print('mean_rew: %f' %np.mean(rew_stat))
		#self.logger.log('reward', rew_stat)
