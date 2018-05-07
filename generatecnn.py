#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = basis vgg16 network
#argv[2] = new fc2 size
#argv[3] = .npy file of train dataset for predictions size
#argv[4] = filename to store the generated Model

from keras.layers import Dense
from keras.models import Model, load_model
import numpy as np
import sys


#Variaveis de entrada e saida da rede
vgg16_imgnet = load_model(sys.argv[1])
fc2_size = int(sys.argv[2])
arr = np.load(sys.argv[3])
n_classes = arr['arr_1'].shape[1]
arr.close()
filename = sys.argv[4]

print("Network: " + sys.argv[1])
print("FC2 Size: " + str(fc2_size))
print("# of classes: " + str(n_classes))
print("Output Network: " + filename)

for layer in vgg16_imgnet.layers:
	layer.trainable = False

new_tensor = Dense(fc2_size, activation='relu', name='fc2')(vgg16_imgnet.get_layer(name='fc1').output)
new_tensor = Dense(n_classes, activation='softmax', name='predictions')(new_tensor)

#Gerar modelo
new_vgg16 = Model(vgg16_imgnet.input, new_tensor)
new_vgg16.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
new_vgg16.summary()

#Salvar modelo
new_vgg16.save(filename)