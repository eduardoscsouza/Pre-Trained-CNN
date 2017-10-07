#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to npz files with arrays
#argv[2] = output .txt filename

from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
import os.path
import sys
import time

in_path = os.path.join("..", "Data", "Datasets", "compressed", "90%", "17flowers")
out_path = os.path.join("..", "Data", "Accuracies", "90%.txt")
if (len(sys.argv) >= 2):
	in_path = sys.argv[1]
if (len(sys.argv) >= 3):
	out_path = sys.argv[2]

print("Input: " + in_path)
print("Output: " + out_path)

out_list = np.load(in_path)
X_list = out_list['arr_0']
Y_list = np.argmax(out_list['arr_1'], axis=1)

init_clock = time.clock()
init_time = time.time()
classifier = svm.SVC()
score = cross_val_score(classifier, X_list, Y_list, cv=10)

out_file = open(out_path, 'a')
print("Processor time in seconds: " + str(time.clock()-init_clock), file=out_file)
print("Real time in seconds: " + str(time.time()-init_time), file=out_file)
print(os.path.split(in_path)[-1], end='\t', file=out_file)
print(np.array_str(score).replace('\n', ''), end="\n\n", file=out_file)
out_file.close()
