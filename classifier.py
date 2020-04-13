import keras
from keras import optimizers

import sys

from data_generator import DataGenerator
from models import Models

class Train(object):
	def __init__(self, shape, nclasses):
		# self.model = Model(shape, nclasses).build_lenet()
		# self.model = Models(shape, nclasses).build_spinenet()
		self.model = Models(shape, nclasses).build_resnet34()

	def train(self, trn, val, batch, epochs):

		lr_rate = 0.01
		decay_rate = lr_rate / epochs
		opt = keras.optimizers.SGD(lr=lr_rate, decay=decay_rate)

		self.model.compile(loss=keras.losses.categorical_crossentropy,
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

def main():
	# input_shape = (32, 32, 3)
	# num_classes = 10
	input_shape = (256, 256, 3)
	num_classes = 2
	batch = 32
	nepochs = 20
	trn_loader = DataGenerator(sys.argv[1], input_shape, batch, True, False)
	val_loader = DataGenerator(sys.argv[2], input_shape, batch, False, False)
	
	trainer = Train(input_shape, num_classes).train(trn_loader, val_loader, batch, nepochs)

if __name__=='__main__':
	main()
