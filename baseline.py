#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to load the images

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from math import ceil
import numpy as np
import os.path
import sys

img_path = os.path.join("..", "Data", "Datasets", "downloaded", "classified", "coil-20")
if (len(sys.argv) >= 2):
	img_path = sys.argv[1]

vgg16_imgnet = VGG16(weights='imagenet', include_top=True)
vgg16_imgnet.summary()

out_fc1 = K.function([vgg16_imgnet.get_layer(index=0).input], [vgg16_imgnet.get_layer(name='fc1').output])
out_fc2 = K.function([vgg16_imgnet.get_layer(index=0).input], [vgg16_imgnet.get_layer(name='fc2').output])

img_batch_size = 256
img_gen = ImageDataGenerator()
img_flow = img_gen.flow_from_directory(img_path, target_size=(224, 224), class_mode='categorical', batch_size=img_batch_size, shuffle=False)

net_batch_size = 10
for i in range(ceil(img_flow.samples/img_batch_size)):
	x, y = img_flow.next()
	remain = img_batch_size
	while remain > 0:
		fc1 = out_fc1([x[(img_batch_size-remain):(img_batch_size-remain)+net_batch_size]])[0]
		fc2 = out_fc2([x[(img_batch_size-remain):(img_batch_size-remain)+net_batch_size]])[0]
		for j in range(fc1.shape[0]):
			print(img_flow.filenames[(i*img_batch_size)+(img_batch_size-remain)+j].split('/'))
		
		remain -= net_batch_size
#print("Images array: ", end="")
#print(x.shape)
#print("Labels array: ", end="")
#print(y.shape)
#np.save(os.path.join(out_path, out_prefix + "images"), x)
#np.save(os.path.join(out_path, out_prefix + "labels"), y)'''