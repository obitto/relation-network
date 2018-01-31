import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.rnn_cell import LSTMCell
import numpy as np
from utils import *

class RN(object):

	def __init__(self, config, is_train = True, restore = False, mode = 'clevr'):
		"""
		model for Relation Augment network.
		
		Args:
			config: object containing configuration parameters.
			is_train: whether this model would be trained.
			restore: whether this model would be initialized from last checkpoint.
			mode: which task this model is used on. There are two option: [clevr, sclevr].
			since the input for this two task is different the structure for these two task
			would be different. And they use different save path.
		"""
		self.config = config
		self.mode = mode
		self.is_train = is_train
		self.rotate_degree = (np.random.rand()- 0.5) / 10
		self.ckpt_path = "model/" + mode + "/" + mode + "_rn.ckpt"

		#build the model
		self.build()
		self.global_step = tf.train.get_or_create_global_step(graph=None)
		
		#define optimizer here
		self.optimizer = tf.contrib.layers.optimize_loss(
			loss=self.loss,
			global_step=self.global_step,
			learning_rate=self.config.learning_rate,
			optimizer=tf.train.AdamOptimizer,
			name='optimizer_loss')

		#define saver
		self.saver = tf.train.Saver()
		
		#start session
		self.session = tf.Session()
		
		#restore and initialize all parameters
		if restore == True:
			self.session.run(tf.global_variables_initializer())
			self.saver.restore(self.session, self.ckpt_path)
		else:
			# Initialize all Variables
			self.session.run(tf.global_variables_initializer())
		
	def get_feed_dict(self, batch_chunk, step=None, is_training=None):
		"""
		function to feed the batch input to each corresponding placeholder.
		"""
		if self.mode == 'clevr':
			fd = {
				self.image: np.vstack([sample['image'] for sample in batch_chunk]),
				self.question: [sample['question'] for sample in batch_chunk],
				self.answer: np.vstack([sample['answer'] for sample in batch_chunk]),
				self.seq_length: [sample['seq_length'] for sample in batch_chunk]
			}
		else:
			fd = {
				self.image: np.vstack([sample['image'] for sample in batch_chunk]),
				self.question: [sample['question'] for sample in batch_chunk],
				self.answer: np.vstack([sample['answer'] for sample in batch_chunk]),
			}
		self.rotate_degree = [sample['rotate'] for sample in batch_chunk]
		return fd
	
	def build(self):
		"""
		function to build all block.
		"""
		self.build_input()
		self.build_CNN()
		self.build_Encoder()
		self.build_Relation()
		self.build_loss()
	
	def build_input(self):
		"""
		build input placeholder.
		for clevr task, the input would be actual quesiton sentence, therefore it requires a LSTM encoder
		to encode the question.
		for sort-of-clevr task, the input is already embedding, there is no need for the encoder.
		"""
		self.image = tf.placeholder(shape = self.config.image_shape,
									dtype = tf.float32,
									name = 'image')
		if self.mode == 'clevr':
			self.question = tf.placeholder(shape = [self.config.batch_size,None],
										dtype = tf.int32,
										name = 'question')
			self.seq_length = tf.placeholder(dtype = tf.int32, shape = [None], name = 'seq_length')
		else:
			self.question = tf.placeholder(shape = [self.config.batch_size,self.config.question_dim],
										dtype = tf.float32,
										name = 'question')
		self.answer = tf.placeholder(shape = [self.config.batch_size, self.config.answer_dim],
									dtype = tf.float32,
									name = 'answer')
	
	def build_CNN(self):
		"""
		build Convolution neural network to embed the image into objects.
		"""
		with tf.variable_scope('Convolution') as scope:
			self.conv1 = conv2d(self.image, self.config.filters, 3, name = 'conv1', padding = "same", is_train = self.is_train)
			self.conv2 = conv2d(self.conv1, self.config.filters, 3, name = 'conv2', padding = "same", is_train = self.is_train)
			self.conv3 = conv2d(self.conv2, self.config.filters, 3, name = 'conv3', padding = "same", is_train = self.is_train)
			self.conv4 = conv2d(self.conv3, self.config.filters, 3, name = 'conv4', padding = "same", is_train = self.is_train)
		
	def build_Encoder(self):
		"""
		if the model is used for clevr, build an LSTM encoder and use the final hidden state as the question embedding.
		"""
		with tf.variable_scope('Encoder') as scope:
			if self.mode == 'clevr':
				#initialize word embedding
				self.embeddings = tf.Variable(tf.random_normal([self.config.vocab_size, self.config.embed_dim], stddev=0.35), name = 'embeddings', trainable = self.is_train)
				embedded = tf.nn.embedding_lookup(self.embeddings, self.question)
				
				#define LSTM encoder
				self.encoder_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_dim,initializer = tf.contrib.layers.xavier_initializer())
				encoder_outputs, encoder_state = tf.nn.dynamic_rnn(self.encoder_cell, embedded,dtype = tf.float32,scope = scope, sequence_length = self.seq_length)
				
				#take the hidden state as question embedding from LSTM-Tuple
				self.question_embed = encoder_state[1]
			else:
				self.question_embed = self.question
			
	def build_Relation(self):
		"""
		Build the relation block for each pair of objects
		"""
		relations = []
		w = self.conv4.get_shape().as_list()[1]
		h = self.conv4.get_shape().as_list()[2]
		# for each combination of pixel in the final feature map, create a relation pair.
		for i in range(w*h):
			o_i = self.conv4[:, int(i / w), int(i % w), :]
			o_i = tag_obj(o_i, i, w)
			for j in range(w*h):
				o_j = self.conv4[:, int(j / w), int(j % w), :]
				#tag the object pair with coordinate
				o_j = tag_obj(o_j, j, w)
				if i == 0 and j == 0:
					relation = g_theta(o_i, o_j, self.question_embed, reuse=False)
				else:
					relation = g_theta(o_i, o_j, self.question_embed, reuse=True)
				relations.append(relation)
		relations = tf.stack(relations, axis=0)
		#sum over the output from g_theta
		self.relations = tf.reduce_sum(relations, axis=0, name='relation')
			

		
	def f_theta(self, relation, scope = 'f_phi'):
		"""
		final MLP block to generate logits.
		"""
		with tf.variable_scope(scope) as scope:
			fc_1 = fc(relation, 256, name='fc_1')
			fc_2 = fc(fc_1, 256, name='fc_2')
			fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=self.is_train, scope='fc_3/')
			fc_3 = fc(fc_2,self.config.answer_dim, activation_fn=None, name='fc_3')
			return fc_3
		
	def build_loss(self):
		"""
		define loss function here. Use cross_entropy loss.
		"""
		
		#output from final MLP layer
		self.logits = self.f_theta(self.relations)
		self.pred = tf.nn.softmax(self.logits)
		
		#calculate loss
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.answer))
		
		#calculate accuracy for this batch
		self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.answer, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
	
	def run_batch(self, batch):
		"""
		run single batch, if is_train is True, the also run an optimizer to update the parameters.
		"""
		feed = self.get_feed_dict(batch)
		if self.is_train == True:
			loss, pred, acc, _ = self.session.run([self.loss, self.pred, self.accuracy, self.optimizer], feed_dict=feed)
			return loss, pred, acc
		else:
			pred, acc = self.session.run([self.pred, self.accuracy], feed_dict=feed)
			return pred, acc

	def save(self):
		#save the model to ckpt path
		save_path = self.saver.save(self.session, self.ckpt_path)    			
