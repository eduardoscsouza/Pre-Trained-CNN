#!/bin/bash

for qual in 1 5 10 25 50 100
do
	find ../DADOS1/esouza/Datasets/extracted/fc2/$qual%/ -type f -name "*rand.npz" -exec bash -c 'python svm.py '"{}"' ../DADOS1/esouza/Results/SVM/'"$qual"'%_rand.txt ../DADOS1/esouza/Results/SVM/'"$qual"'%_rand.npz' \; > ../DADOS1/esouza/Logs/SVM/svm_rand/$qual%_log.txt 2> ../DADOS1/esouza/Logs/SVM/svm_rand/$qual%_errlog.txt &
done
wait