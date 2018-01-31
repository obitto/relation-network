import tensorflow as tf
import numpy as np
from RelationNetwork import RN
from prepare import SClevrDataset,ClevrDataset
from utils import Config, Config_SClevr
import argparse

class Evaluator(object):
	"""
	This would evaluate some basic metric of the model
	"""
	def __init__(self, config):
		"""
		use different model and checkpoint path fo
		"""
		self.mode = config.model
		if self.mode == 'clevr':
			self.config = Config()
			self.dataset = Clevrdataset(self.config, dataset_name('val'), load_vocab = False)
			self.model = RN(self.config, is_train = False, restore = True, mode = 'clevr')
		else:
			self.config = Config_SClevr()
			self.dataset = SClevrDataset(self.config, name = 'val', shuffle = False)
			self.model = RN(self.config, is_train = False, restore = True, mode = 'sclevr')
	
	def evaluate(self):
		if self.mode == 'clevr':
			self.evaluate_clevr()
		else:
			self.evaluate_sclevr()

	def evaluate_clevr(self):
		print('not finish yet')
		
	def evaluate_sclevr(self):
		finish = 0
		pred = []
		correct_pred = np.zeros((2,3),dtype = np.float32)
		num_question = np.zeros((2,3), dtype = np.float32)
		while finish < 1:
			p,a = self.model.run_batch(self.dataset.next_batch(self.config.batch_size))
			pred.extend(np.argmax(p, axis = 1))
			if self.dataset.counter < self.config.batch_size:
				finish += 1
		pred = pred[:-self.dataset.counter or None]
		if len(pred) != len(self.dataset.questions):
			print("length not match")
		for i in range(len(pred)):
			if self.dataset.questions[i]['question'][6] == 1.0:
				idx = 0
			else:
				idx = 1
			if pred[i] == np.argmax(self.dataset.questions[i]['answer']):
				correct_pred[idx,self.dataset.questions[i]['question'][8:].index(1.0)] += 1
			num_question[idx,self.dataset.questions[i]['question'][8:].index(1.0)] += 1
		print(correct_pred)
		print(num_question)
		print ('Overall accuracy :', float(np.sum(correct_pred)/ float(np.sum(num_question))))
		print ('Accuracy for relation question: ', float(np.sum(correct_pred[0,:]))/np.sum(num_question[0,:]))
		print ('Accuracy for None-relation question: ', float(np.sum(correct_pred[1,:]))/np.sum(num_question[1,:]))
		print ('Accuracy for question shape of the farthest boject for <Color> is ', correct_pred[0,0]/ num_question[0,0])
		print ('Accuracy for question shape of the nearest boject for <Color> is ', correct_pred[0,1]/ num_question[0,1])
		print ('Accuracy for question How many objects have the same shape as object with <Color>', correct_pred[0,2]/ num_question[0,2])
		print ('Accuracy for question Is the <Color> object a circle or a rectangle', correct_pred[1,0]/ num_question[1,0])
		print ('Accuracy for question Is the <Color> object on the top of the image?', correct_pred[1,1]/ num_question[1,1])
		print ('Accuracy for question Is the <Color> object on the right of the image?', correct_pred[1,2]/ num_question[1,2])
			
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='sclevr', choices=['clevr', 'sclevr'])
	config = parser.parse_args()
    
	evaluator = Evaluator(config)
	evaluator.evaluate()
	
if __name__ == '__main__':
	main()
