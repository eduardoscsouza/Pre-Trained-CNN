#!/bin/bash

for per in {10..90..20}
do
	find ../DADOS1/esouza/Datasets/extracted/fc2/$per%/ -type f -name "*rand.npz" -exec bash -c 'python svm.py '"{}"' ../DADOS1/esouza/Results/SVM/'"$per"'%_rand.txt ../DADOS1/esouza/Results/SVM/'"$per"'%_rand.npz' \; > ../DADOS1/esouza/Logs/SVM/svm_rand/$per%_log.txt 2> ../DADOS1/esouza/Logs/SVM/svm_rand/$per%_errlog.txt &
done
find ../DADOS1/esouza/Datasets/extracted/fc2/uncompressed/ -type f -name "*rand.npz" -exec bash -c 'python svm.py '"{}"' ../DADOS1/esouza/Results/SVM/uncompressed_rand.txt ../DADOS1/esouza/Results/SVM/uncompressed_rand.npz' \; > ../DADOS1/esouza/Logs/SVM/svm_rand/uncompressed_log.txt 2> ../DADOS1/esouza/Logs/SVM/svm_rand/uncompressed_errlog.txt &
wait