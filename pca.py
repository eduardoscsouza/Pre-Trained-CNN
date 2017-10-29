#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to .npz files with arrays
#argv[2] = output .npz filename
#argv[3] = output size

from sklearn.decomposition import PCA
import numpy as np
import os.path
import sys

in_path = os.path.join("..", "Data", "Datasets", "compressed", "90%", "17flowers")
out_path = os.path.join("..", "Data", "Accuracies", "17flowers")
out_size = 4096
if (len(sys.argv) >= 2):
	in_path = sys.argv[1]
if (len(sys.argv) >= 3):
	out_path = sys.argv[2]
if (len(sys.argv) >= 4):
	out_size = int(sys.argv[3])

print("Input: " + in_path)
print("Output: " + out_path)
print("Outsize: " + str(out_size))

in_list = np.load(in_path)
X_list = in_list['arr_0']
Y_list = in_list['arr_1']

pca = PCA(n_components=out_size)
out_X_list = pca.fit_transform(X_list)

np.savez_compressed(out_path, out_X_list, Y_list)

print("Output X Shape: " + str(out_X_list.shape))
print("Output Y Shape: " + str(Y_list.shape))
