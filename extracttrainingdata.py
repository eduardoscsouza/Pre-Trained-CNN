#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to load input images
#argv[2] = path to save output arrays
#argv[3] = percentage of dataset to use as training data

from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import numpy as np
import os.path
import sys

#Argumentos
img_path = os.path.join("..", "Data", "Datasets", "compressed", "90%", "17flowers")
out_path = os.path.join("..", "Data", "Datasets", "extracted", "90%", "17flowers_imgnet")
per = 0.05
if (len(sys.argv) >= 2):
	img_path = sys.argv[1]
if (len(sys.argv) >= 3):
	out_path = sys.argv[2]
if (len(sys.argv) >= 4):
	per = int(sys.argv[3]) / 100.0

print("Input: " + img_path)
print("Output: " + out_path)
print("Percentage: " + str(100*per) + "%")

#Criar um objeto que cria um iterador que percorre e classifica as imagens
#baseado na estrutura de pastas
img_batch_size = 256
img_gen = ImageDataGenerator()
img_flow = img_gen.flow_from_directory(img_path, target_size=(224, 224), class_mode='categorical', batch_size=img_batch_size, shuffle=True)

#Gerar vetores finais com todas as imagens e labels
per_class = int(per*(img_flow.samples//img_flow.num_class))
n = int(per_class*img_flow.num_class)
out_idx = 0
out_X = np.empty(shape=(n, 224, 224, 3))
out_Y = np.empty(shape=(n, img_flow.num_class))
class_count = np.zeros(img_flow.num_class)
while(np.min(class_count) < per_class):
	aux_x, aux_y = img_flow.next()
	for i in range(aux_x.shape[0]):
		c = np.argmax(aux_y[i])
		if (class_count[c] < per_class):
			out_X[out_idx] = aux_x[i]
			out_Y[out_idx] = aux_y[i]
			class_count[c] += 1
			out_idx += 1

np.savez_compressed(out_path, out_X, out_Y)

print("Output X Shape: " + str(out_X.shape))
print("Output Y Shape: " + str(out_Y.shape))