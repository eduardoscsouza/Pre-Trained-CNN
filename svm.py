#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to npz files with arrays
#argv[2] = output .txt filename
#argv[3] = output .npz filename

from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
import os.path
import sys

in_path = os.path.join("..", "Data", "Datasets", "compressed", "90%", "17flowers")
out_path = os.path.join("..", "Data", "Accuracies", "90%.txt")
out_npz_path = os.path.join("..", "Data", "Accuracies", "90%")
if (len(sys.argv) >= 2):
	in_path = sys.argv[1]
if (len(sys.argv) >= 3):
	out_path = sys.argv[2]
if (len(sys.argv) >= 4):
	out_npz_path = sys.argv[3]

print("Input: " + in_path)
print("Output: " + out_path)
print("Output .npz: " + out_npz_path)

in_list = np.load(in_path)
X_list = in_list['arr_0']
Y_list = np.argmax(in_list['arr_1'], axis=1)
in_list.close()

shuff = np.random.permutation(X_list.shape[0])
shuff_X_list, shuff_Y_list = X_list[shuff], Y_list[shuff]

classifier = svm.SVC()
score = cross_val_score(classifier, shuff_X_list, shuff_Y_list, cv=10)

data_name = os.path.split(in_path)[-1]
out_file = open(out_path, 'a')
print(data_name, end='\t', file=out_file)
print(np.array_str(score).replace('\n', ''), file=out_file)
out_file.close()

exists = True
in_npz = np.empty((0, 10))
try:
	in_npz = np.load(out_npz_path)
except IOError:
	exists = False

out_dict = {data_name:score}
if exists:
	for file in in_npz:
		if (not file in out_dict):
			out_dict[file] = in_npz[file]
	in_npz.close()
np.savez_compressed(out_npz_path, **out_dict)

print("Number of scores in .npz file: " + str(len(out_dict)))	