
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50

class Models(object):
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

	def build_spinenet(self):
		#img_input = Input(shape=(self.input_shape[1], self.input_shape[0], self.input_shape[2]))
		x = Conv2D(96, (3, 3), strides=1, padding='same', name='conv1')(img_input)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling2D()(x)
		x = Conv2D(256, (3, 3), strides=1, padding='same', name='conv2')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling2D()(x)
		x = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='conv3')(x)
		x = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='conv4')(x)
		x = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='conv5')(x)
		x = MaxPooling2D()(x)
		x = Flatten(name='flatten')(x)
		x = Dense(512, activation='relu', name='fc6')(x)
		x = Dense(512, activation='relu', name='fc7')(x)
		x = Dense(self.output_shape, activation='sigmoid', name='fc8')(x)
		# x = Dense(self.output_shape, activation='softmax', name='fc8')(x)
		inputs = img_input
		model = Model(inputs, x, name='spine')
		return model


	def build_resnet34(self):
		model = ResNet50(include_top=False)
		x = model.output
		x = Dropout(0.5)(x)
		x = GlobalAveragePooling2D()(x)
		# output = Dense(self.output_shape, activation='sigmoid')(x)
		output = Dense(self.output_shape, activation='softmax')(x)
		return Model(inputs=model.input, outputs=output)
