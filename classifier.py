import keras
from keras import optimizers

import os
import sys
import cv2

from data_generator import DataGenerator
from models import Models

class Classifier(object):
	def __init__(self, shape, nclasses, model_path=''):
		self.model = Models(shape, nclasses).build_alexnet_red(last_act='sigmoid')
		self.loader = DataGenerator('', shape)
		if os.path.exists(model_path):
			self.model = Models(shape, nclasses).load_model(model_path)

	def train(self, trn, val, batch, epochs):

		lr_rate = 0.01
		decay_rate = lr_rate / epochs
		opt = keras.optimizers.SGD(lr=lr_rate, decay=decay_rate)

		self.model.compile(loss=keras.losses.binary_crossentropy,
		              optimizer=opt,
		              metrics=['accuracy'])

		self.model.fit_generator(trn,
					steps_per_epoch=int(trn.n_samples/batch),
					epochs=epochs,
					callbacks=None,
					validation_data=val,
					validation_steps=int(val.n_samples/batch),
					class_weight=None,
					workers=1,
					initial_epoch=0)

		self.model.save('final_classifier.h5')

	def predict(self, image, thresh=0.5):
		image = self.loader.preprocess(image, expand=True)
		pred = self.model.predict(image)[0]
		pred = [1 if p > thresh else 0 for p in pred
		return pred

def main():
	input_shape = (128, 128, 3)
	num_classes = 2
	batch = 32
	nepochs = 50
	if sys.argv[-1] == '-train':
		trn_loader = DataGenerator(sys.argv[1], input_shape, batch, True, False)
		val_loader = DataGenerator(sys.argv[2], input_shape, batch, False, False)
		trainer = Classifier(input_shape, num_classes).train(trn_loader, val_loader, batch, nepochs)
	else:
		classifier = Classifier(input_shape, num_classes, sys.argv[1])
		for files in os.listdir(sys.argv[2]):
			filename = os.path.join(sys.argv[2], files)
			img = cv2.imread(filename)
			pred = classifier.predict(img)
			print (pred)
			x = input()

if __name__=='__main__':
	main()
