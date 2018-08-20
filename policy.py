import tensorflow as tf 
import numpy as np
from sklearn.utils import shuffle
import pickle

class Policy(object):
	def __init__(self, obs_dim, act_dim, max_kl, init_logvar, 
				 epochs, logger, clip=0):
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		self.init_logvar = init_logvar
		self.epochs = epochs
		self.logger = logger

		self.lr_mult = 1.0
		self.clip = clip
		# d_kl related, beta and eta is NOT passed as an argument
		self.beta = 1.0
		self.eta = 50.0
		self.kl_target = max_kl

		self.g = tf.Graph()
		with self.g.as_default():
			self._placeholders()
			self._nn_policy()
			self._loss_train_op()
			self.init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()

		# session
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49, allow_growth=True)
		self.sess = tf.Session(graph= self.g, config=tf.ConfigProto(
							   gpu_options=gpu_options, allow_soft_placement=True))
		self.sess.run(self.init)

	def _placeholders(self):
		# observations and actions
		self.policy_obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'policy_obs')
		self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
		# old means and log_vars
		self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')
		self.old_logvars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_logvars')
		# advantages
		self.adv_ph = tf.placeholder(tf.float32, (None,), 'adv')
		# learning rate for adam
		self.policy_lr_ph = tf.placeholder(tf.float32, (), 'policy_lr')
		# d_kl loss strength terms
		self.beta_ph = tf.placeholder(tf.float32, (), 'beta')

	def _nn_policy(self):
		hid1_size = self.obs_dim * 10
		hid3_size = self.act_dim * 10
		hid2_size = int(np.sqrt(hid1_size * hid3_size))
		self.lr = 9e-4 / np.sqrt(hid2_size)
		out = tf.layers.dense(self.policy_obs_ph, hid1_size, tf.tanh,
		                      kernel_initializer=tf.random_normal_initializer(
		                          stddev=np.sqrt(1 / self.obs_dim)), name="h1")
		out = tf.layers.dense(out, hid2_size, tf.tanh,
		                      kernel_initializer=tf.random_normal_initializer(
		                          stddev=np.sqrt(1 / hid1_size)), name="h2")
		out = tf.layers.dense(out, hid3_size, tf.tanh,
		                      kernel_initializer=tf.random_normal_initializer(
		                          stddev=np.sqrt(1 / hid2_size)), name="h3")
		self.means = tf.layers.dense(out, self.act_dim,
		                             kernel_initializer=tf.random_normal_initializer(
		                                 stddev=np.sqrt(1 / hid3_size)), name="means")
		logvar_speed = (10 * hid3_size) // 48
		log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
		                           tf.constant_initializer(0.0))
		self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.init_logvar

		# sample 
		self.sampled_act = (self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(self.act_dim,)))
		if self.clip:
			self.sampled_act = tf.clip_by_value(sampled_act, -1.0, 1.0)
	
	def _loss_train_op(self):
		# necessary calculations for calculating losses
		# log probabilities
		logp = -0.5 * tf.reduce_sum(self.log_vars) - 0.5 * tf.reduce_sum((tf.square(self.act_ph - self.means) / tf.exp(self.log_vars)), axis=1)
		old_logp = -0.5 * tf.reduce_sum(self.old_logvars_ph) - 0.5 * tf.reduce_sum((tf.square(self.act_ph - self.old_means_ph) / tf.exp(self.old_logvars_ph)), axis=1)

		# mean_kl
		log_det_cov = tf.reduce_sum(self.log_vars)
		log_det_cov_old = tf.reduce_sum(self.old_logvars_ph)
		trace = tf.reduce_sum(tf.exp(self.old_logvars_ph - self.log_vars))
		term = tf.reduce_sum(tf.square(self.old_means_ph - self.means) / tf.exp(self.log_vars), axis=1)

		self.mean_kl = 0.5 * tf.reduce_mean(trace + term + log_det_cov - log_det_cov_old - self.act_dim)
		self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) + tf.reduce_sum(self.log_vars))

		# losses
		loss1 = -tf.reduce_mean(tf.exp(logp - old_logp) * self.adv_ph)
		loss2 = tf.reduce_mean(self.beta_ph * self.mean_kl)
		loss3 = self.eta * tf.square(tf.maximum(0.0, self.mean_kl - 2 * self.kl_target))
		
		self.policy_loss = loss1 + loss2 + loss3

		# optimizer
		policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.policy_lr_ph)
		self.policy_minimizer = policy_optimizer.minimize(self.policy_loss)

	def sample(self, obs):
		acts = self.sess.run(self.sampled_act, feed_dict={self.policy_obs_ph: obs})
		return acts

	def update(self, observes, actions, advantages):
		# this shuffle is ADDED
		# i will also ADD batch_updates LATER
		feed_dict = {self.policy_obs_ph: observes, 
					 self.act_ph: actions,
					 self.adv_ph: advantages,
					 self.policy_lr_ph: self.lr * self.lr_mult,
					 self.beta_ph: self.beta}
		old_means, old_logvars = self.sess.run([self.means, self.log_vars], feed_dict=feed_dict)
		feed_dict[self.old_means_ph] = old_means
		feed_dict[self.old_logvars_ph] = old_logvars

		# for some epochs update theta (policy)
		# policy update
		kl = 0
		ent = []
		for _ in range(self.epochs):
			self.sess.run(self.policy_minimizer, feed_dict=feed_dict)
			kl, entropy = self.sess.run([self.mean_kl, self.entropy], feed_dict=feed_dict)
			ent.append(entropy)
			if kl > self.kl_target * 4:
				break
		
		self.logger.log('policy_entropy', ent)
		print('mean_ent: %f' %np.mean(ent))

		# we will PLAY with this later
		if kl > 2 * self.kl_target:
			self.beta = np.minimum(35, 1.5*self.beta)
			if self.beta > 30 and self.lr_mult > 0.1:
				self.lr_mult /= 1.5
		elif kl < self.kl_target / 2:
			self.beta = np.maximum(1 / 35, self.beta/1.5)
			if self.beta < 1 / 30 and self.lr_mult < 10:
				self.lr_mult *= 1.5
		
	def save_session(self, filename):
		self.saver.save(self.sess, filename)

	def restore_session(self, session_to_restore, stats_to_recover=None):
		self.saver.restore(self.sess, session_to_restore)
		if stats_to_recover is not None:
			fileobject = open(stats_to_recover, 'rb')
			stats = pickle.load(fileobject)
			fileobject.close()
			print(np.max(stats['trew_stat']))
			name = 'std_mean'
			for key, value in enumerate(stats):
				if value == 'scale_offset':
					name = 'scale_offset'
			return stats[name][-1][0], stats[name][-1][1]

	def close_session(self):
		self.sess.close()