#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to load the images
#argv[2] = file to save output matrix
#argv[3] = layer name
#argv[3] = bool to load imagnet weights

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from math import ceil
import numpy as np
import os.path
import sys

img_path = os.path.join("..", "Data", "Datasets", "compressed", "90%", "17flowers")
out_path = os.path.join("..", "Data", "Datasets", "extracted", "90%", "17flowers_fc2.npy")
layer_name = 'fc2'
imagenet = True
if (len(sys.argv) >= 2):
	img_path = sys.argv[1]
if (len(sys.argv) >= 3):
	out_path = sys.argv[2]
if (len(sys.argv) >= 4):
	layer_name = sys.argv[3]
if (len(sys.argv) >= 5):
	imagenet = (sys.argv[4].lower() in ('yes', 'true', 't', 'y', '1'))

print("Input: " + img_path)
print("Output: " + out_path)
print("Layer: " + layer_name)
print("Imagenet Weights: ", end='')
print(imagenet)

vgg16_imgnet = VGG16(weights=None, include_top=True)
if imagenet:
	vgg16_imgnet = VGG16(weights='imagenet', include_top=True)
vgg16_imgnet.summary()

out_layer = K.function([vgg16_imgnet.get_layer(index=0).input], [vgg16_imgnet.get_layer(name=layer_name).output])

img_batch_size = 256
img_gen = ImageDataGenerator()
img_flow = img_gen.flow_from_directory(img_path, target_size=(224, 224), class_mode='categorical', batch_size=img_batch_size, shuffle=False)

out_X_list = np.empty(shape=(0, vgg16_imgnet.get_layer(name=layer_name).output.shape[1]))
out_Y_list = np.empty(shape=(0, img_flow.num_class))
net_batch_size = 8
for i in range(ceil(img_flow.samples/img_batch_size)):
	x, y = img_flow.next()
	out_Y_list = np.concatenate((out_Y_list, y), axis=0)
	remain = img_batch_size
	while remain > 0:
		aux = out_layer([x[(img_batch_size-remain):(img_batch_size-remain)+net_batch_size]])[0]
		out_X_list = np.concatenate((out_X_list, np.asarray(aux)), axis=0)
		remain -= net_batch_size

np.savez(out_path, out_X_list, out_Y_list)