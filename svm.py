#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to load the images
#argv[2] = .txt filename

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn import svm
from sklearn.model_selection import cross_val_score
from math import ceil
import numpy as np
import os.path
import sys

img_path = os.path.join("..", "Data", "Datasets", "compressed", "90%")
out_path = os.path.join("..", "Data", "Accuracies", "90%.txt")
if (len(sys.argv) >= 2):
	img_path = sys.argv[1]
if (len(sys.argv) >= 3):
	out_path = sys.argv[2]

vgg16_imgnet = VGG16(weights='imagenet', include_top=True)
vgg16_imgnet.summary()

out_fc2 = K.function([vgg16_imgnet.get_layer(index=0).input], [vgg16_imgnet.get_layer(name='fc2').output])

img_batch_size = 256
img_gen = ImageDataGenerator()
root, dirs, files = next(os.walk(img_path))
out_file = open(out_path, 'w')
for cur_dir in dirs:
	cur_path = os.path.join(img_path, cur_dir)
	img_flow = img_gen.flow_from_directory(cur_path, target_size=(224, 224), class_mode='categorical', batch_size=img_batch_size, shuffle=False)

	fc2_out_list = np.empty(shape=(0, 4096))
	Y_list = np.empty(shape=(0))
	net_batch_size = 10
	for i in range(ceil(img_flow.samples/img_batch_size)):
		x, y = img_flow.next()
		Y_list = np.concatenate((Y_list, np.argmax(y, axis=1)), axis=0)
		remain = img_batch_size
		while remain > 0:
			aux = out_fc2([x[(img_batch_size-remain):(img_batch_size-remain)+net_batch_size]])[0]
			fc2_out_list = np.concatenate((fc2_out_list, np.asarray(aux)), axis=0)
			remain -= net_batch_size

	classifier = svm.SVC()
	score = cross_val_score(classifier, fc2_out_list, Y_list, cv=10)
	#print(cur_dir, end='\t', file=out_file, )
	#print(score, file=out_file)
