import numpy as np
import json
from nltk.tokenize import word_tokenize
from PIL import Image
import os
import h5py
from random import shuffle

class ClevrDataset(object):
	"""
	This is the dataset for clevr task.
	"""
	
	def __init__(self, config, dataset_name = 'train' , shuffle = False, load_vocab = False):
		if dataset_name == 'train':
			self.question_path = 'data/CLEVR_v1.0/questions/CLEVR_train_questions.json'
		elif dataset_name == 'val':
			self.question_path = 'data/CLEVR_v1.0/questions/CLEVR_val_questions.json'
		elif dataset_name == 'test':
			self.question_path = 'data/CLEVR_v1.0/questions/CLEVR_test_questions.json'   
		
		#hard coded answer index
		self.answerSet = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
			 '5': 5, '6': 6, '7': 7, '8': 8,'9':9, '10': 10, 
			 'cyan':11, 'brown': 12, 'metal':13, 'cube': 14, 'purple': 15, 'green': 16,
			 'large':17, 'cylinder': 18, 'no': 19, 'blue': 20, 'yellow': 21, 'sphere': 22, 
			 'red': 23, 'rubber': 24, 'yes': 25, 'gray': 26, 'small': 27}
		self.questions = self.load_QA_data(self.question_path)
		if load_vocab == False:
			self.build_vocab()
			self.save_vocab()
		else:
			self.load_vocab()
		
		#max sentence length
		self.max_length = 50
		self.images = self.load_image(self.questions, dataset_name)
		self.counter = 0
		self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
		self.std = np.array([0.229, 0.224, 0.224]).reshape(1, 1, 1, 3)
		if shuffle == True:
			shuffle(self.questions)
		
	def tokenize(self, sentence, token_to_discard = ['?','.',',']):
		"""
		tokenize the word, and discard some certain tokens by requirement
		"""
		tokens = word_tokenize(sentence)
		res = []
		for token in tokens:
			if token not in token_to_discard:
				res.append(token)
		return res
		
	def build_vocab(self, min_count = 1, token_to_discard = []):
		"""
		build word2idx and idx2word vocabulary.
		
		Args:
			min_count : minimum occurrence of word to be included in the vocabulary.
			token_to_discard: a list of token would be discard from the input sentence.
		"""
		token_count = {}
		self.word2idx = {}
		tokens = [self.tokenize(sentence[0], token_to_discard) for sentence in self.questions]
		for i in range(len(tokens)):
			self.questions[i] = [tokens[i], self.questions[i][1], self.questions[i][2]]
			for word in tokens[i]:
				if word not in token_count:
					token_count[word] = 1
				else:
					token_count[word] = token_count[word] + 1
		
		# add special 
		self.word2idx['unk'] = 0
		self.word2idx['pad'] = 1
		
		# extract word appear above the threshold
		for word in token_count.keys():
			if token_count[word] >= min_count:
				self.word2idx[word] = len(self.word2idx)
		# create idx to word dictionary
		self.idx2word = {v: k for k, v in self.word2idx.items()}

	def save_vocab(self, file = 'model/CLEVR_vocab.json'):
		with open(file, 'w') as fp:
			json.dump(self.word2idx, fp)
			
	def load_vocab(self, file = 'model/CLEVR_vocab.json'):
		with open(file, 'w') as fp:
			self.word2idx = json.load(fp)
			self.idx2word = {v: k for k, v in self.word2idx.items()}
			
	def pad(self, question):
		"""
		pad the input question to desired length with token <pad>
		"""
		if (len(question) > self.max_length):
			print("max lenght exceeded: ", len(question))
		while(len(question) < self.max_length):
			question.append(1)
		return question
	
	def convert2idx(self, sentence):
		"""
		convert sentence into index, so it can be feed into LSTM encoder
		"""
		idx = []
		for word in sentence:
			if word in self.word2idx.keys():
				idx.append(self.word2idx[word])
			else:
				idx.append(0)
		return idx
	
	def load_QA_data(self, path, max_sample = None):
		"""
		load the question and answers.
		"""
		with open(path) as f:
			data = json.load(f)
		if max_sample != None:
			questions = [(sample['question'], sample['answer'], sample['image_filename']) for sample in data['questions'][:max_sample]]
		else:
			questions = [(sample['question'], sample['answer'], sample['image_filename']) for sample in data['questions']]
		return questions

	def toOneHot(self, answer):
		"""
		convert answer to one hot.
		"""
		idx = self.answerSet[answer]
		one_hot = np.zeros((1, len(self.answerSet)))
		one_hot[0][idx] = 1.0
		return one_hot
	
	def load_image(self, questions, path):
		"""
		load the image, resize it to desired size, then normalize it
		"""
		prefix = 'data/CLEVR_v1.0/images/'
		images = {}
		for question in questions:
			if question[2] not in images:
				im = Image.open(prefix + path + '/' + questions[0][2])
				im = im.convert('RGB')
				im = im.resize((80, 80))
				im = np.array(im)
				images[question[2]] = im
		return images
	
	def create_sample_tuple(self, question, answer, image):
		"""
		create one sample
		"""
		sample = {
			'question': self.pad(self.convert2idx(question)),
			'answer': self.toOneHot(answer),
			'image': image,
			'seq_length': len(question),
			'rotate': ( np.random.random_sample()- 0.5) / 10
		}
		return sample
	
	def next_batch(self, batch_size):
		"""
		return a batch of data samples
		"""
		if (self.counter + batch_size) < len(self.questions):
			batch = [self.create_sample_tuple(question[0], question[1], self.prepocess(self.images[question[2]])) for question in self.questions[self.counter:self.counter+batch_size]]
			self.counter += batch_size
		else:
			batch = [self.create_sample_tuple(question[0], question[1], self.prepocess(self.images[question[2]])) for question in self.questions[self.counter:]]
			self.counter = self.counter + batch_size - len(self.questions)
			batch.extend([self.create_sample_tuple(question[0], question[1], self.prepocess(self.images[question[2]])) for question in self.questions[:self.counter]])
		return batch
	 
	def prepocess(self, image):
		return (image / 255.0 - self.mean) / self.std

class SClevrDataset(object):
	"""
	This is the dataset for sort-of-clevr task.
	"""

	def __init__(self, config, name = 'train',shuffle = True):
		self.path = 'data/' + name +'.json'
		self.name = name
		self.counter = 0
		self.load_data()
		
	def load_data(self):
		"""
		load the dataset from file
		"""
		self.images = {}
		with open(self.path) as f:
			data = json.load(f)
		self.questions = data['qa']

		#shuffle it if necessary
		if shuffle == True:
			shuffle(self.questions)
		for image in data['image']:
			self.images[image['id']] = np.array(image['image'])
	
	def get_data(self, question):
		""" 
		preprocessing and data augmentation
		"""
		idx = question['id']
		img = self.images[idx]/255.0
		q = np.array(question['question'], dtype = np.float32)
		a = np.array(question['answer'], dtype = np.float32)
		sample = {
			'question': q,
			'answer': a,
			'image': np.expand_dims(img, axis = 0),
			'rotate': ( np.random.random_sample()- 0.5) / 10
		}
		return sample
	
	def next_batch(self, batch_size):
		"""
		return a batch of data samples
		"""
		if (self.counter + batch_size) < len(self.questions):
			batch = [self.get_data(question) for question in self.questions[self.counter:(self.counter + batch_size)]]
			self.counter += batch_size
		else:
			batch = [self.get_data(question) for question in self.questions[self.counter:]]
			self.counter = self.counter + batch_size - len(self.questions)
			if shuffle == True:
				shuffle(self.questions)
			batch.extend([self.get_data(question) for question in self.questions[:self.counter]])
		return batch
	
