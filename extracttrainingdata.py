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
n = int(per*img_flow.samples)
out_X = np.empty(shape=(ceil(n/img_batch_size)*img_batch_size, 224, 224, 3))
out_Y = np.empty(shape=(ceil(n/img_batch_size)*img_batch_size, img_flow.num_class))
for i in range(ceil(n/img_batch_size)):
	aux_x, aux_y = img_flow.next()
	out_X[i*img_batch_size:(i*img_batch_size)+aux_x.shape[0]] = aux_x
	out_Y[i*img_batch_size:(i*img_batch_size)+aux_y.shape[0]] = aux_y

out_X, out_Y = out_X[:n], out_Y[:n]

np.savez_compressed(out_path, out_X, out_Y)

print("Output X Shape: " + str(out_X.shape))
print("Output Y Shape: " + str(out_Y.shape))