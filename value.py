import tensorflow as tf 
import numpy as np
from sklearn.utils import shuffle

class Value(object):
	def __init__(self, obs_dim, act_dim, epochs, batch_size, logger):
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		self.epochs = epochs
		self.batch_size = batch_size

		# logger
		self.logger = logger

		# replay buffer
		self.replay_obs = None
		self.replay_rews = None

		self.g = tf.Graph()
		with self.g.as_default():
			self._placeholders()
			self._nn_value()
			self._loss_train_op()
			self.init = tf.global_variables_initializer()
		
		# session
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49, allow_growth=True)
		self.sess = tf.Session(graph= self.g, config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
		self.sess.run(self.init)

	def _placeholders(self):
		self.value_obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'value_obs')
		self.disc_rew_ph = tf.placeholder(tf.float32, (None,), 'disc_rew')

	def _nn_value(self):
		hid1_size = self.obs_dim * 10
		hid3_size = 5
		hid2_size = int(np.sqrt(hid1_size * hid3_size))
		self.lr = 1e-2 / np.sqrt(hid2_size)
		out = tf.layers.dense(self.value_obs_ph, hid1_size, tf.tanh,
		                      kernel_initializer=tf.random_normal_initializer(
		                          stddev=np.sqrt(1 / self.obs_dim)), name="h1")
		out = tf.layers.dense(out, hid2_size, tf.tanh,
		                      kernel_initializer=tf.random_normal_initializer(
		                          stddev=np.sqrt(1 / hid1_size)), name="h2")
		out = tf.layers.dense(out, hid3_size, tf.tanh,
		                      kernel_initializer=tf.random_normal_initializer(
		                          stddev=np.sqrt(1 / hid2_size)), name="h3")
		out = tf.layers.dense(out, 1,
		                      kernel_initializer=tf.random_normal_initializer(
		                          stddev=np.sqrt(1 / hid3_size)), name='output')
		self.value = tf.squeeze(out)

	def _loss_train_op(self):
		# loss
		self.value_loss = tf.reduce_mean(tf.square(self.disc_rew_ph - self.value))

		# optimizer
		value_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.value_minimizer = value_optimizer.minimize(self.value_loss)

	def predict(self, obs):
		vpred = self.sess.run(self.value, feed_dict={self.value_obs_ph: obs})
		return vpred

	def update(self, obs, rews):
		if self.replay_obs is not None:
			train_obs = np.concatenate([obs, self.replay_obs])
			train_rews = np.concatenate([rews, self.replay_rews])
		else:
			train_obs = obs
			train_rews = rews

		self.replay_obs = obs
		self.replay_rews = rews

		ep_loss = []
		for _ in range(self.epochs):
			obs, rews = shuffle(train_obs, train_rews)
			n_batches = max(obs.shape[0] // self.batch_size, 1)
			bobs = np.array_split(obs, n_batches, axis=0)
			brews = np.array_split(rews, n_batches)
			batch_loss = []
			for i in range(n_batches):
				feed_dict = {self.value_obs_ph: bobs[i],
							 self.disc_rew_ph: brews[i]}
				self.sess.run(self.value_minimizer, feed_dict=feed_dict)
				loss = self.sess.run(self.value_loss, feed_dict=feed_dict)
				batch_loss.append(loss)
			ep_loss.append(np.mean(batch_loss))
			
	def close_session(self):
		self.sess.close()