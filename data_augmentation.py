import cv2
import numpy as np

class DataAugmentation(object):
	def __init__(self, aug_list=['rot', 'scl', 'sft', 'blr']):
		self.aug_list = aug_list

	def rotate(self, image):
		cen = (image.shape[1]/2, image.shape[0]/2)
		deg = np.random.uniform(-30, 30)
		mat = cv2.getRotationMatrix2D(cen, deg, 1.0)
		rotated = cv2.warpAffine(image, mat, image.shape[1::-1])
		return rotated

	def scale(self, image):
		sc = np.random.uniform(0.8, 1.2)
		H, W = image.shape[:2]
		h, w = int(H*sc), int(W*sc)
		image = cv2.resize(image, (w, h))
		x, y = int(abs(H-h)/2), int(abs(W-w)/2)
		if sc < 1.0:
			scaled = np.zeros((H, W, 3), dtype=np.int8)
			scaled[y:y+h, x:x+w] = image
		else:
			scaled = image[y:y+h, x:x+w]
			scaled = cv2.resize(scaled, (W, H))
		return scaled

	def shift(self, image):
		H, W = image.shape[:2]
		sh = np.random.randint(0, 4)
		f = np.random.uniform(0.0, 0.2)
		x, y = int(W*f), int(H*f)
		shifted = np.zeros((H, W, 3), dtype=np.int8)
		if sh == 0: # left
			shifted[0:H, 0:W-x] = image[0:H, x:W]
		elif sh == 1: # right
			shifted[0:H, x:W] = image[0:H, 0:W-x]
		elif sh == 2: # up
			shifted[0:H-y, 0:W] = image[y:H, 0:W]
		else: # down
			shifted[y:H, 0:W] = image[0:H-y, 0:W]
		return shifted

	def blur(self, image):
		blurred = cv2.blur(image, (3, 3))
		return blurred

	def flip(self, image, d):
		flipped = cv2.flip(image, d)
		return flipped

	def augment_one(self, image):
		rn = np.random.randint(0, len(self.aug_list))
		if rn == 0:
			return self.rotate(image)
		elif rn == 1:
			return self.scale(image)
		elif rn == 2:
			return self.shift(image)
		else:
			return self.blur(image)

	def augment(self, image):
		augmented = []
		for aug in self.aug_list:
			if aug == 'rot':
				augmented.append(self.rotate(image))
			elif aug == 'scl':
				augmented.append(self.scale(image))
			elif aug == 'sft':
				augmented.append(self.shift(image))
			elif aug == 'blr':
				augmented.append(self.blur(image))
			elif aug == 'flh':
				augmented.append(self.flip(image, 1))
			elif aug == 'flv':
				augmented.append(self.flip(image, 0))
			return augmented
