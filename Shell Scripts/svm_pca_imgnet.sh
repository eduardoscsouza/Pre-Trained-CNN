#!/bin/bash

for pow in {3..10}
do
	aux_pow=$((2**$pow))
	for qual in 1 5 10 25 50 100
	do
		find ../DADOS1/esouza/Datasets/reduced/fc2/$aux_pow/$qual/ -type f -name "*imgnet.npz" -exec bash -c 'python svm.py '"{}"' ../DADOS1/esouza/Results/SVM_PCA/'"$aux_pow"'/'"$qual"'_imgnet.txt ../DADOS1/esouza/Results/SVM_PCA/'"$aux_pow"'/'"$qual"'_imgnet.npz' \; > ../DADOS1/esouza/Logs/SVM_PCA/svm_pca_imgnet/$aux_pow/$qual%_log.txt 2> ../DADOS1/esouza/Logs/SVM_PCA/svm_pca_imgnet/$aux_pow/$qual%_errlog.txt &
	done
	wait
done