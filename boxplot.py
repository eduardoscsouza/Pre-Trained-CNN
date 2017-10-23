#Author: Eduardo Santos Carlos de Souza

#Usage:
#argv[1] = path to .npz files with arrays
#argv[2] = output .pdf filename

import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys

in_path = os.path.join("..", "Data", "Datasets", "compressed", "90%", "17flowers")
out_path = os.path.join("..", "Data", "Accuracies", "90%.txt")
if (len(sys.argv) >= 2):
	in_path = sys.argv[1]
if (len(sys.argv) >= 3):
	out_path = sys.argv[2]

print("Input: " + in_path)
print("Output: " + out_path)

names = []
in_list = np.load(in_path)
for file in in_list:
	names.insert(0, file)

idx = 0
scores = np.empty((5, 10))
names.sort()
for name in names:
	scores[idx] = in_list[name]
	idx = idx + 1
in_list.close()

cur_fig = plt.figure(figsize=(8, 10))
cur_fig.canvas.set_window_title("Dataset's Scores")

plt.boxplot(np.transpose(scores))
plt.title("Dataset's Scores")
plt.xlabel("Dataset")
plt.ylabel("Score")
plt.xticks(range(1, len(names)+1), names, rotation='vertical')
plt.tight_layout()
plt.savefig(out_path)
plt.show()

print(names)
print(scores.shape)
print(scores)