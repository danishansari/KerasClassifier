
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

class Models(object):
	def __init__(self, inp_shape, out_shape):
		self.input_shape = inp_shape
		self.output_shape = out_shape

	def load_model(self, path):
		return load_model(path)

	def build_lenet(self):
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.output_shape, activation='softmax'))
		return model

	def build_alexnet_red(self, last_act='softmax'):
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(5, 5), strides=2, activation='relu', input_shape=self.input_shape))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=1))
		model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=1))
		model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
		model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
		model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=1))
		model.add(Flatten())
		model.add(Dense(256, activation='relu'))
		model.add(Dense(256, activation='relu'))
		model.add(Dense(256, activation='relu'))
		model.add(Dense(self.output_shape, activation=last_act))
		return model
