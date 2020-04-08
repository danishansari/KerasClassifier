# data generator code ------------

from keras.utils import Sequence

import os, sys
import numpy as np
import cv2
from collections import defaultdict

from data_augmentation import DataAugmentation

class DataGenerator(Sequence):
	def __init__(self, path, shape, batch, shuffle, augment):
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

	def get_sample(self, index):
		fname, label = self.list_ids[index]
		fpath = os.path.join(self.path, label)
		fpath = os.path.join(fpath, fname)
		image = cv2.imread(fpath)
		image = cv2.resize(image, self.shape[:2])
		if self.augment:
			image = self.augment.augment_one(image)
		image = image / 255.
		label = self.one_hot(label)
		return image, label
	
	def one_hot(self, label):
		oh_label = np.zeros((self.n_classes))
		oh_label[int(label)] = 1.
		return oh_label

	def load(self):
		filepath = os.path.dirname(self.path)
		list_ids, count = [], 0
		self.class_weight = defaultdict(int)
		with open(self.path) as fp:
			data = fp.readlines()
		for d in data:
			fname = d.strip().split(',')[0]
			fname = os.path.basename(fname)
			label = d.strip().split(',')[1]
			list_ids.append([fname, label])
			self.class_weight[label] += 1 
			count +=1 
		self.path = filepath
		self.classes = list(self.class_weight.keys())
		return list_ids, count, len(self.classes)
