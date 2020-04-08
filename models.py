
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

class Model(object):
	def __init__(self, inp_shape, out_shape):
		self.input_shape = inp_shape
		self.output_shape = out_shape

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
