# data generator code ------------

from keras.utils import Sequence

import os, sys
import numpy as np
import cv2
from collections import defaultdict

from data_augmentation import DataAugmentation

class DataGenerator(Sequence):
	def __init__(self, path, shape, batch=1, shuffle=False, augment=False):
		self.path = path
		self.shape = shape
		self.batch = batch
		self.shuffle = shuffle
		self.list_ids, self.n_samples, self.n_classes = self.load()
		self.on_epoch_end()
		self.augment = False
		if augment:
			self.augment = DataAugmentation()

	def __len__(self):
		return int(self.n_samples/self.batch)

	def __getitem__(self, index):
		tmp_indexes = self.indexes[index*self.batch:(index+1)*self.batch]
		X = np.empty((self.batch, self.shape[0], self.shape[1], self.shape[2]))
		y = np.empty((self.batch, self.n_classes))
		for i in range(len(tmp_indexes)):
			X[i,], y[i,] = self.get_sample(tmp_indexes[i])
		return X, y

	def on_epoch_end(self):
		self.indexes = np.arange(self.n_samples)
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def preprocess(self, image, expand=False):
		image = cv2.resize(image, self.shape[:2])
		if self.augment:
			image = self.augment.augment_one(image)
		image = image.astype('float32') / 255.
		if expand:
			image = np.expand_dims(image, axis=0)
		return image

	def get_sample(self, index):
		fname, label = self.list_ids[index]
		fpath = os.path.join(self.path, fname)
		image = cv2.imread(fpath)
		image = self.preprocess(image)
		label = self.one_hot(label)
		return image, label
	
	def one_hot(self, labels):
		oh_label = np.zeros((self.n_classes))
		for lab in labels:
			if len(lab) > 0:
				oh_label[self.classes.index(lab)] = 1.
		return oh_label

	def load(self):
		if os.path.exists(self.path):
			filepath = os.path.dirname(self.path)
			list_ids, count = [], 0
			self.class_weight = defaultdict(int)
			with open(self.path) as fp:
				data = fp.readlines()
			for d in data:
				fname = d.strip().split(',')[1]
				labels = d.strip().split(',')[2:]
				list_ids.append([fname, labels])
				for lab in labels:
					if len(lab) > 0:
						self.class_weight[lab] += 1 
				count +=1 
			self.path = filepath
			self.classes = list(self.class_weight.keys())
			print ('classes:', self.classes, self.class_weight)
			return list_ids, count, len(self.classes)
		else:
			return [], 0, 0
