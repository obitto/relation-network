import h5py
import numpy as np
from PIL import Image, ImageDraw
import os
import argparse
from PIL import Image, ImageDraw
import json
import argparse

COLOR = [
	(0, 0, 210),   #red
	(0, 210, 0),   #green
	(210, 0, 0),   #blue
	(150, 150, 0), #yellow
	(150, 0, 150), #magenta
	(0, 150, 150), #cyan
]

class SClevr_generator(object):
	
	def __init__(self, config):
		self.config = config
		self.question_dim = 11
		self.answer_dim = 10
		self.path = 'data/' + config.path + '.json'
		
	def generate_dataset(self):
		"""
		generate a dataset with defined dataset size and image size.
		Then dump it as json.
		first generate an image with an id. Then randomly generate 10 relation question-answer and 
		10 non-relation question-answer based on the image with same id.
		"""
		img_set = []
		qa_set = []
		for i in range(self.config.dataset_size):
			img, r = self.generate_image()
			q = self.generate_question()
			a = self.generate_answer(r, q)
			img_sample = {
				'id': i,
				'image': img.tolist()
			}
			img_set.append(img_sample)
			for j in range(len(q)):
				qa_sample = {
					'id': i,
					'question': q[j].tolist(),
					'answer': a[j].tolist()
				}
				qa_set.append(qa_sample)
		print('Finished creating smaples')
		dataset = {
			'image':	img_set,
			'qa':	qa_set
		}
		with open(self.path, "w") as f:
			json.dump(dataset, f)
	
	  
	def generate_centers(self):
		"""
		generate center of objects
		make sure each center is valid and there is no collision between centers.
		"""
		centers = []
		size = self.config.image_size
		for i in range(self.config.num_obj):
			flag = True
			while flag:
				c = np.random.randint(int(size * 0.05), int(size * 0.95), 2)
				flag = False
				for center in centers:
					if (abs(center[0] - c[0]) <= 0.1 * size) or (abs(center[1] - c[1]) <= 0.1 *size):
						flag = False
			centers.append(c)
				
		return centers
	
	def generate_image(self):
		"""
		generate one image, with predefined number of objects.
		each object(circle or square), has radius 0.05 * image_size.
		each object has different color.
		no collision.
		right now only generate square images.
		"""
		centers = self.generate_centers()
		img = Image.new('RGB', (self.config.image_size, self.config.image_size), color=(0,0,0))
		shapes = np.random.randint(2, size=len(centers))
		drawer = ImageDraw.Draw(img)
		r = int(0.05 * self.config.image_size)
		R = []
		for i in range(len(centers)):
			coor = (centers[i][0] - r , centers[i][1] - r, centers[i][0] + r, centers[i][1] + r)
			if shapes[i] < 0.5:
				drawer.rectangle(coor, fill=COLOR[i])
			else:
				drawer.ellipse(coor, fill=COLOR[i])
			R.append([centers[i], i, shapes[i]])
		return np.array(img), R
	
	def generate_question(self, num_question = 10):
		"""
		generate question representation, the default dimension for this is 11.
		where the question looks like 
		[blue, green, red, yellow, magenta, cyan, relation, non-relation, question1, question2, question3]
		where only one of the color would be tagged as 1, that's the base object in question
		the question can be either relation or non-relation, and for each type there are three question.
		This is a binary representation of question embedding
		"""
		
		questions = []
		for q in range(num_question):
			for r in range(2):
				question = np.zeros(self.question_dim, dtype = np.float32)
				color = np.random.randint(len(COLOR))
				question[color] = 1.0
				question[6 + r] = 1.0
				question_label = np.random.randint(3)
				question[8 + question_label] = 1.0
				questions.append(question)
		return questions
	
	def generate_answer(self, R, questions):
		#[yes, no, square, circle, 1, 2, 3, 4, 5, 6]
		"""
		generate answer based on image representation and question embedding
		
		Args:
			R: representation of the image, it has information of the centers and shapes for each object.
			questions: a list of question embedding
		Return:
			answers: a list of answer in one hot form.
			each answer is in the form of [yes, no, square, circle, 1, 2, 3, 4, 5, 6]
		"""
		answers = []
		for question in questions:
			color = np.where(question[:6] == 1.0)[0][0]
			answer = np.zeros(self.answer_dim, dtype = np.float32)
			if question [6] == 1.0:
				if question[8]: #The shape of the nearest object?
					dist = [((R[color][0]-obj[0])**2).sum() for obj in R]
					dist[dist.index(0)] = float('inf')
					closest = dist.index(min(dist))
					if R[closest][2] < 0.5:
						answer[2] = 1
					else:
						answer[3] = 1
				elif question[9]: #The shape of the farthest object?
					dist = [((R[color][0]-obj[0])**2).sum() for obj in R]
					furthest = dist.index(max(dist))
					if R[furthest][2] < 0.5:
						answer[2] = 1
					else:
						answer[3] = 1

				else: #How many objects have the same shape?
					count = -1
					shape = R[color][2]
					for obj in R:
						if obj[2] == shape:
							count += 1
					answer[count + 4] = 1.0
			else:
				if question[8]: #Is it a circle or a rectangle?
					if R[color][2] < 0.5:
						answer[2] = 1
					else:
						answer[3] = 1
				elif question[9]: #Is it on the top of the image?
					if R[color][0][1] < self.config.image_size/2:
						answer[0] = 1
					else:
						answer[1] = 1
				else: #Is it on the right of the image?
					if R[color][0][0] < self.config.image_size/2:
						answer[1] = 1
					else:
						answer[0] = 1
			answers.append(answer)
		return answers
		
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_size', type=int, default=10000)
	parser.add_argument('--image_size', type=int, default=80)
	parser.add_argument('--num_obj', type=int, default=6)
	parser.add_argument('--split_rate', type=float, default=2e-2)
	config = parser.parse_args()
	
	config.path = 'train'
	size = config.dataset_size
	config.dataset_size = int(size * (1 - config.split_rate))
	gen_train = SClevr_generator(config)
	gen_train.generate_dataset()
	config.path = 'val'
	config.dataset_size = int(size * config.split_rate)
	gen_val = SClevr_generator(config)
	gen_val.generate_dataset()
	
	
	
if __name__ == '__main__':
	main()
		
