from __future__ import division
import cv2
import os
import numpy as np
import tensorflow as tf
import utils
import logging


# img = cv2.imread("a.JPEG")
# imgblack = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)
# print(imgblack.shape)
FLAGS = tf.app.flags.FLAGS

def get_image_list(train_dir = './train2014/'):
	image_list = []
	index = 0
	for dir in os.listdir(train_dir):
		index += 1
		image_list.append(os.path.abspath(train_dir + dir))
		if (index > 50000):
			break
	length = len(image_list)
	return image_list[: -FLAGS.test_size], image_list[-FLAGS.test_size:]
	#return image_list[: int(length)], image_list[int(length * 3 / 4):]

def load_image(image_dir, size):
	img = cv2.imread(image_dir)
	img = cv2.resize(img, (size, size))
	imglab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
	imgblack = imglab[:, :, 0]
	return imglab / 255, imgblack / 255
	#return img, imgblack

def naive_encode(image, size):
	image_resize = cv2.resize(image, (int(size / 4), int(size / 4)))
	image_return = np.zeros([image_resize.shape[0], image_resize.shape[1], 256])
	for i in range(0, image_resize.shape[0]):
		for j in range(0, image_resize.shape[1]):
			# temp_a = int(image_resize[i][j][0] * 16)
			# temp_b = int(image_resize[i][j][1] * 16)
			# point = [[temp_a - 1, temp_b], [temp_a + 1, temp_b], [temp_a, temp_b - 1], [temp_a, temp_b + 1], [temp_a, temp_b]]
			# point_sum = 0
			# for p in point:
			# 	if (p[0] >= 0 and p[0] < 16 and p[1] >= 0 and p[1] < 16):
			# 		image_return[i][j][p[0] * 16 + p[1]] = gaussian_kernel(p[0], p[1], image_resize[i][j][0] * 16, image_resize[i][j][1] * 16)
			# 		point_sum += image_return[i][j][p[0] * 16 + p[1]]
			# for p in point:
			# 	image_return[i][j][p[0] * 16 + p[1]] /= point_sum
			# 	#print(image_return[i][j][p[0] * 16 + p[1]])
			temp_a = int(image_resize[i][j][0] * 16)
			temp_b = int(image_resize[i][j][1] * 16)
			image_return[i][j][temp_a * 16 + temp_b] = 1
	return image_return

def naive_decode(image, size, T = 0.38):
	image_show = np.zeros([image.shape[0], image.shape[1], 2])
	denominator = np.sum(np.power(image, 1 / T), axis = 2)
	denominator = np.reshape(denominator, [64, 64, 1])
	pos_dist = np.power(image, 1 / T) / denominator
	progression = np.arange(image.shape[2])
	#print(progression)
	image_show[:, :, 0] = np.sum(pos_dist * (progression // 16 + 0.5) / 16, axis = 2)
	image_show[:, :, 1] = np.sum(pos_dist * (progression % 16 + 0.5) / 16, axis = 2)
	# for i in range(0, image.shape[0]):
	# 	for j in range(0, image.shape[1]):
	# 		for k in range(0, image.shape[2]):
	# 			image_show[i][j][0] += np.power(image[i][j][k], 1 / T) / denominator[i][j] * (k // 16 + 0.5) / 16
	# 			image_show[i][j][1] += np.power(image[i][j][k], 1 / T) / denominator[i][j] * (k % 16 + 0.5) / 16
	image_show = cv2.resize(image_show, (size, size))
	return image_show

def for_display(black, rg, output, dir):
	if not os.path.exists(dir):
		os.makedirs(dir)
	output_reshape = np.reshape(output, [FLAGS.picture_size, FLAGS.picture_size, 2])
	output_con = np.zeros([FLAGS.picture_size, FLAGS.picture_size, 3])
	output_con[:, :, 0] = black
	output_con[:, :, 1:] = output_reshape
	save_LABimage(output_con , dir + "/output_color.jpg")
	output_con[:, :, 0] = 0.5
	save_LABimage(output_con , dir + "/ab.jpg")
	real_con = np.zeros([FLAGS.picture_size, FLAGS.picture_size, 3])
	real_con[:, :, 0] = black
	real_con[:, :, 1:] = rg
	save_LABimage(real_con , dir + "/real_color.jpg")
	save_image(black , dir + "/black_white.jpg")
	logging.info(dir)
	logging.info(utils.psnr(output_con * 255, real_con * 255)) 

def save_image(image, image_dir):
	image_temp = image * 255
	cv2.imwrite(image_dir, image_temp)

def save_LABimage(image, image_dir):
	image_temp = (image * 255).astype(np.dtype('uint8'))
	image_RGB = cv2.cvtColor(image_temp, cv2.COLOR_LAB2RGB)
	#print(image_RGB)
	cv2.imwrite(image_dir, image_RGB)

def gaussian_kernel(x, y, xc, yc, sigma = 0.5):
	return np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * (sigma ** 2)))