import tensorflow as tf 
import numpy as np 
from utils import Scaler
from sklearn.utils import shuffle

class Discriminator(object):
	def __init__(self, obs_dim, act_dim, ent_reg_weight, epochs, 
				 input_type, loss_type, logger):
		self.obs_dim = obs_dim
		self.act_dim = act_dim

		self.input_type = input_type
		self.loss_type = loss_type
		if self.input_type == 'states_actions':
			self.input_dim = obs_dim + act_dim
		elif self.input_type == 'states':
			self.input_dim = obs_dim

		self.epochs = epochs

		# we are only NORMALIZING states for now
		self.scaler = Scaler(self.obs_dim)

		# SET LEARNING RATE
		self.lr_mult = 1.0
		self.ent_reg_weight = ent_reg_weight

		# logger
		self.logger = logger

		# creating graph
		self.g = tf.Graph()
		with self.g.as_default():
			self._placeholders()
			self._nn_disc()
			self._loss_train_op()
			self.init = tf.global_variables_initializer()

		# session
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49, allow_growth=True)
		self.sess = tf.Session(graph= self.g, config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
		self.sess.run(self.init)

	def _placeholders(self):
		self.input_ph = tf.placeholder(tf.float32, (None, self.input_dim), name='inputs')
		self.labels_ph = tf.placeholder(tf.float32, (None,), name='labels')
		self.weights_ph = tf.placeholder(tf.float32, (None,), name='weights')
		self.lr_ph = tf.placeholder(tf.float32, (), name='learning_rate')

	def _nn_disc(self):
		hid1_size = 300
		hid2_size = 200
		self.lr = 1e-4

		'''
		hid1_size = self.obs_dim * 10
		hid3_size = self.act_dim * 10
		hid2_size = int(np.sqrt(hid1_size * hid3_size))
		self.lr = 9e-4 / np.sqrt(hid2_size)
		'''
		out = tf.layers.dense(self.input_ph, hid1_size, tf.tanh,
		                      kernel_initializer=tf.random_normal_initializer(
		                          stddev=np.sqrt(1 / self.obs_dim)), name="h1")
		out = tf.layers.dense(out, hid2_size, tf.tanh,
		                      kernel_initializer=tf.random_normal_initializer(
		                          stddev=np.sqrt(1 / hid1_size)), name="h2")
		'''
		out = tf.layers.dense(out, hid3_size, tf.tanh,
		                      kernel_initializer=tf.random_normal_initializer(
		                          stddev=np.sqrt(1 / hid2_size)), name="h3")
		'''

		scores = tf.layers.dense(out, 1, tf.identity,
		                             kernel_initializer=tf.random_normal_initializer(
		                                 stddev=np.sqrt(1 / hid2_size)), name="scores")

		self.scores = tf.squeeze(scores)

		# rewards could be clipped
		self.reward_op = -tf.log(1 - tf.nn.sigmoid(self.scores))

	def _loss_train_op(self):
		if self.loss_type == 'pure_gail':
			cross_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.labels_ph)
			# this extra entropy penalty is NOT included in the paper
			# taken from the example provided by the authors
			# in the paper there is an entropy term for TRPO update???
			ent_loss = (1.0 - tf.nn.sigmoid(self.scores)) * self.scores + tf.nn.softplus(-self.scores)
			self.loss = tf.reduce_mean((cross_loss - self.ent_reg_weight * ent_loss) * self.weights_ph)	

			train_op = tf.train.AdamOptimizer(learning_rate=self.lr_ph)
			self.train_min = train_op.minimize(self.loss)
		elif self.loss_type == 'wasserstein':
			self.loss = tf.reduce_mean((self.labels_ph * self.scores + (1.0 - self.labels_ph) * self.scores) * self.weights_ph)
			train_op = tf.train.AdamOptimizer(learning_rate=self.lr_ph)
			self.train_min = train_op.minimize(self.loss)

	def normalize_input(self, inpt):
		# check out this normalization
		self.scaler.update(inpt)
		# i was getting in reverse order		
		scale, offset = self.scaler.get()
		inpt = (inpt - offset) * scale 
		return inpt

	def get_rewards(self, gen_obs, gen_acts=None):
		# those observations are already normalized
		scale, offset = self.scaler.get()
		gen_obs = (gen_obs - offset) * scale
		gen_input = gen_obs
		if self.input_type == 'states_actions':
			gen_input = np.concatenate([gen_obs, gen_acts], axis=1)
		return self.sess.run(self.reward_op, feed_dict={self.input_ph: gen_input}) 

	def update(self, exp_obs, gen_obs):
		# shuffle generator observations and actions
		gen_obs = shuffle(gen_obs)

		obs = np.concatenate([gen_obs, exp_obs], axis=0)
		obs = self.normalize_input(obs)

		# number of generator examples
		gen_num = gen_obs.shape[0]
		exp_num = exp_obs.shape[0]
		
		# create labels and mark real/fake
		labels = np.zeros((gen_num + exp_num))
		labels[gen_num:] = 1.0

		# calc loss weight
		weights = np.zeros((gen_num + exp_num))
		weights[:gen_num] = gen_num / (gen_num + exp_num)
		weights[gen_num:] = exp_num / (gen_num + exp_num)

		for i in range(self.epochs):
			inpt, labels, weights = shuffle(obs, labels,  weights)
			bobs = np.array_split(inpt, self.epochs, axis=0)
			blabs = np.array_split(labels, self.epochs)
			bweg = np.array_split(weights, self.epochs)
			for j in range(self.epochs):
				loss, _ = self.sess.run([self.loss, self.train_min], 
										feed_dict={self.input_ph: bobs[i],
												   self.labels_ph: blabs[i],
												   self.weights_ph: bweg[i],
												   self.lr_ph: self.lr * self.lr_mult})

		# evaluate the discriminator
		scores = self.sess.run(self.scores, feed_dict={self.input_ph: obs})

		def sigmoid(x):
			return 1 / (1 + np.exp(-x))
		gen_corr = np.sum((sigmoid(scores[:gen_num]) < 0.5))
		exp_corr = np.sum((sigmoid(scores[gen_num:]) > 0.5))
		gen_acc = gen_corr / gen_num
		exp_acc = exp_corr / exp_num
		total_acc = (gen_corr + exp_corr) / (gen_num + exp_num)

		# log necessary info
		#self.logger.log('gen_acc', gen_acc)
		#self.logger.log('exp_acc', exp_acc)
		#self.logger.log('total_acc', total_acc)
		return gen_acc, exp_acc, total_acc

	def close_session(self):
		self.sess.close()