#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to load the images
#argv[2] = path to save the .txt resulting files

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from math import ceil
import numpy as np
import os.path
import sys

img_path = os.path.join("..", "Data", "Datasets", "downloaded", "classified")
out_path = os.path.join("..", "Data", "Baselines")
if (len(sys.argv) >= 2):
	img_path = sys.argv[1]
if (len(sys.argv) >= 3):
	out_path = sys.argv[2]

vgg16_imgnet = VGG16(weights='imagenet', include_top=True)
vgg16_imgnet.summary()

out_fc1 = K.function([vgg16_imgnet.get_layer(index=0).input], [vgg16_imgnet.get_layer(name='fc1').output])
out_fc2 = K.function([vgg16_imgnet.get_layer(index=0).input], [vgg16_imgnet.get_layer(name='fc2').output])

img_batch_size = 256
img_gen = ImageDataGenerator()
img_flow = img_gen.flow_from_directory(img_path, target_size=(224, 224), class_mode='categorical', batch_size=img_batch_size, shuffle=False)

fc1_list = []
fc2_list = []
net_batch_size = 10
for i in range(ceil(img_flow.samples/img_batch_size)):
	x, y = img_flow.next()
	remain = img_batch_size
	while remain > 0:
		aux_fc1 = out_fc1([x[(img_batch_size-remain):(img_batch_size-remain)+net_batch_size]])[0]
		aux_fc2 = out_fc2([x[(img_batch_size-remain):(img_batch_size-remain)+net_batch_size]])[0]
		for j in range(aux_fc1.shape[0]):
			label, name = img_flow.filenames[(i*img_batch_size)+(img_batch_size-remain)+j].split('/')
			fc1_list.append((label, name, aux_fc1[j]))
			fc2_list.append((label, name, aux_fc2[j]))
		
		remain -= net_batch_size

fc1_list = sorted(fc1_list)
fc2_list = sorted(fc2_list)

np.set_printoptions(threshold=np.nan, linewidth=np.nan)
fc1_file = open(out_path + "fc1_output.txt", 'w')
fc2_file = open(out_path + "fc2_output.txt", 'w')
for i in range(len(fc1_list)):
	print(fc1_list[i][1], end=" ", file=fc1_file)
	print(fc1_list[i][2], end=" ", file=fc1_file)
	print(fc1_list[i][0], file=fc1_file)

	print(fc2_list[i][1], end=" ", file=fc2_file)
	print(fc2_list[i][2], end=" ", file=fc2_file)
	print(fc2_list[i][0], file=fc2_file)
