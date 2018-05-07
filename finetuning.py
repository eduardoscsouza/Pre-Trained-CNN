#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = filename to cnn
#argv[2] = filename to numpy array with images and labels
#argv[3] = filename to save the trained network
#(optional) argv[4] = batch size
#(optional) argv[5] = epoch size
#(optional) argv[6] = threshold

from keras import backend
from keras.models import load_model
import numpy as np
import sys

cnn = load_model(sys.argv[1])
arr = np.load(sys.argv[2])
images = arr['arr_0']
labels = arr['arr_1']
out_filename = sys.argv[3]
batch_size = 128
if (len(sys.argv) >= 5):
	batch_size = int(argv[4])
epoch_size = 5
if (len(sys.argv) >= 6):
	batch_size = int(argv[5])
threshold = 0.0001
if (len(sys.argv) >= 6):
	threshold = float(argv[6])

print("CNN: " + sys.argv[1])
print("Dataset: " + sys.argv[2])
print("Output Filename: " + sys.argv[3])
print("Batch Size: " + str(batch_size))
print("Epoch Size: " + str(epoch_size))

epoch_count = 0
cur_loss = 100000000
last_loss = 200000000
while(np.abs(cur_loss-last_loss) >= threshold):
	last_loss = cur_loss
	hist = cnn.fit(images, labels, epochs=epoch_size, batch_size=batch_size, verbose=False)
	cur_loss = hist.history['loss'][-1]
	epoch_count += epoch_size
	print("Loss at epoch " + str(epoch_count) + ": " + str(cur_loss))
	print("Accuracy at epoch " + str(epoch_count) + ": " + str(hist.history['acc'][-1]))
cnn.summary()

cnn.save(out_filename)
backend.clear_session()
arr.close()
