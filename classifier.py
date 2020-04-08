import keras
from keras import optimizers

import sys

from data_generator import DataGenerator
from models import Model

class Train(object):
	def __init__(self, shape, nclasses):
		self.model = Model(shape, nclasses).build_lenet()

	def train(self, trn, val, batch, epochs):
		self.model.compile(loss=keras.losses.categorical_crossentropy,
		              optimizer=keras.optimizers.Adadelta(),
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

def main():
	input_shape = (32, 32, 3)
	num_classes = 10
	batch = 128
	nepochs = 20
	trn_loader = DataGenerator(sys.argv[1], input_shape, batch, True, False)
	val_loader = DataGenerator(sys.argv[2], input_shape, batch, False, False)
	
	trainer = Train(input_shape, num_classes).train(trn_loader, val_loader, batch, nepochs)

if __name__=='__main__':
	main()
