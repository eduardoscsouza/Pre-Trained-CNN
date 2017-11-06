#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to directory containing .npz files with arrays. directory must contain subdirectories SVM and SVM_PCA. The SVM directory should have the arrays,
#names starting by their compression quality, and the SVM_PCA directory should have 1 subdirectory for each PCA compression, eacho of those being similat to SVM.
#argv[2] = dataset name
#argv[3] = PCA compression level
#argv[4] = string to filer the npz files
#argv[5] = output .pdf filename

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import fnmatch

in_path = os.path.join("..", "Data", "Datasets", "compressed", "90%", "17flowers")
data_name = "17_flowers.npz"
pca_compression = 4096
name_filter = "*"
out_path = os.path.join("..", "Data", "Accuracies", "90%.pdf")
if (len(sys.argv) >= 2):
	in_path = sys.argv[1]
if (len(sys.argv) >= 3):
	data_name = sys.argv[2]
if (len(sys.argv) >= 4):
	pca_compression = int(sys.argv[3])
if (len(sys.argv) >= 5):
	name_filter = sys.argv[4]
if (len(sys.argv) >= 6):
	out_path = sys.argv[5]

print("Input: " + in_path)
print("Dataset: " + data_name)
print("PCA compression: " + str(pca_compression))
print("Filter: " + name_filter)
print("Output: " + out_path)

if (pca_compression == 4096):
	in_path = os.path.join(in_path, "SVM")
else:
	in_path = os.path.join(in_path, "SVM_PCA", str(pca_compression))

files_dict = dict()
for file in os.listdir(in_path):
	if fnmatch.fnmatch(file, name_filter):
		files_dict[int(str(file).split('_')[0])] = file

qualities = sorted(files_dict)
scores = np.empty((0, 10))
for qual in qualities:
	aux_file = np.load(os.path.join(in_path, files_dict[qual]))
	scores = np.vstack((scores, aux_file[data_name]))
	aux_file.close()

cur_fig = plt.figure(figsize=(8, 10))
cur_fig.canvas.set_window_title("Dataset's Scores")

plt.boxplot(np.transpose(scores))
plt.title("Dataset = " + data_name + " | # of dimensions = " + str(pca_compression))
plt.xlabel("Quality")
plt.ylabel("Score")
plt.xticks(range(1, len(qualities)+1), qualities, rotation='vertical')
plt.tight_layout()
plt.savefig(out_path)

print(scores.shape)
print(scores)