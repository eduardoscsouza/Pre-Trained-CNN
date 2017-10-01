#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to load the images
#argv[2] = .txt filename
#argv[3] = bool to load imagnet weights

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn import svm
from sklearn.model_selection import cross_val_score
from math import ceil
import numpy as np
import os.path
import sys
import time

img_path = os.path.join("..", "Data", "Datasets", "compressed", "90%", "17flowers")
out_path = os.path.join("..", "Data", "Accuracies", "90%.txt")
imagenet = True
if (len(sys.argv) >= 2):
	img_path = sys.argv[1]
if (len(sys.argv) >= 3):
	out_path = sys.argv[2]
if (len(sys.argv) >= 4):
	imagenet = (sys.argv[3].lower() in ('yes', 'true', 't', 'y', '1'))

print("Input: " + img_path)
print("Output: " + out_path)
print("Imagenet Weights: ", end='')
print(imagenet)

vgg16_imgnet = VGG16(weights=None, include_top=True)
if imagenet:
	vgg16_imgnet = VGG16(weights='imagenet', include_top=True)
vgg16_imgnet.summary()

out_fc2 = K.function([vgg16_imgnet.get_layer(index=0).input], [vgg16_imgnet.get_layer(name='fc2').output])

img_batch_size = 256
img_gen = ImageDataGenerator()
img_flow = img_gen.flow_from_directory(img_path, target_size=(224, 224), class_mode='categorical', batch_size=img_batch_size, shuffle=False)

fc2_out_list = np.empty(shape=(0, 4096))
Y_list = np.empty(shape=(0))
net_batch_size = 32
for i in range(ceil(img_flow.samples/img_batch_size)):
	x, y = img_flow.next()
	Y_list = np.concatenate((Y_list, np.argmax(y, axis=1)), axis=0)
	remain = img_batch_size
	while remain > 0:
		aux = out_fc2([x[(img_batch_size-remain):(img_batch_size-remain)+net_batch_size]])[0]
		fc2_out_list = np.concatenate((fc2_out_list, np.asarray(aux)), axis=0)
		remain -= net_batch_size

init_clock = time.clock()
init_time = time.time()
classifier = svm.SVC()
score = cross_val_score(classifier, fc2_out_list, Y_list, cv=10)

out_file = open(out_path, 'a')
print("Processor time in seconds: " + str(time.clock()-init_clock), file=out_file)
print("Real time in seconds: " + str(time.time()-init_time), file=out_file)
print(cur_dir, end='\t', file=out_file)
print(score, end="\n\n", file=out_file)
out_file.close()
