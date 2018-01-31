import tensorflow as tf
import numpy as np
from RelationNetwork import RN
from prepare import SClevrDataset,ClevrDataset
from utils import Config, Config_SClevr
import argparse
import sys

def str2bool(s):
	if s == 'true':
		return True
	else:
		return False
class Trainer(object):

	def __init__(self, config):
		"""
		Trainer to train the model.
		based on the configuration, the model could be either for Clevr task or Sort-of-Clevr task.
		"""
		self.restore = str2bool(config.restore)
		self.mode = config.model
		if self.mode == 'clevr':
			self.config = Config()
			self.dataset = Clevrdataset(self.config, dataset_name('train'), load_vocab = False)
			self.model = RN(self.config, is_train = True, restore = self.restore, mode = 'clevr')
		else:
			self.config = Config_SClevr()
			self.dataset = SClevrDataset(self.config)
			self.model = RN(self.config, is_train = True, restore = self.restore, mode = 'sclevr')
		
	def train(self):
		"""
		Train the model until the max_iter is reached.
		"""
		loss_history = []
		acc_history = []
		loss = [0]
		acc = [0]
		num_epoch = 0
		for i in range(self.config.max_iter):
			if self.dataset.counter < self.config.batch_size:
				sys.stdout.write('\n')
				sys.stdout.flush()
				if num_epoch > 0:
					loss_history.append(np.mean(loss))
					acc_history.append(np.mean(acc))
				num_epoch += 1
				loss = []
				acc = []
			l, p, a  = self.model.run_batch(self.dataset.next_batch(self.config.batch_size))
			loss.append(l)
			acc.append(a)
			sys.stdout.write('\rEpoch: {}, Progress: {} / {}, Loss: {}, Acc: {}'.format(num_epoch, self.dataset.counter, len(self.dataset.questions), str(np.mean(loss)), str(np.mean(acc))))
			sys.stdout.flush()
			if(i % self.config.save == 0):
				self.model.save()
				
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='sclevr', choices=['clevr', 'sclevr'])
	parser.add_argument('--restore', type=str, default='false', choices=['true', 'false'])
	config = parser.parse_args()
	
	trainer = Trainer(config)
	trainer.train()	
if __name__ == '__main__':
	main()
