#!/bin/bash

for pow in {0..10}
do
	aux_pow=$((2**$pow))
	for per in {10..90..20}
	do
		find ../DADOS1/esouza/Datasets/reduced/fc2/$aux_pow/$per%/ -type f -name "*rand.npz" -exec bash -c 'python svm.py '"{}"' ../DADOS1/esouza/Results/SVM_PCA/'"$aux_pow"'/'"$per"'%_rand.txt ../DADOS1/esouza/Results/SVM_PCA/'"$aux_pow"'/'"$per"'%_rand.npz' \; > ../DADOS1/esouza/Logs/SVM_PCA/svm_pca_rand/$aux_pow/$per%_log.txt 2> ../DADOS1/esouza/Logs/SVM_PCA/svm_pca_rand/$aux_pow/$per%_errlog.txt &
	done
	find ../DADOS1/esouza/Datasets/reduced/fc2/$aux_pow/uncompressed/ -type f -name "*rand.npz" -exec bash -c 'python svm.py '"{}"' ../DADOS1/esouza/Results/SVM_PCA/'"$aux_pow"'/uncompressed_rand.txt ../DADOS1/esouza/Results/SVM_PCA/'"$aux_pow"'/uncompressed_rand.npz' \; > ../DADOS1/esouza/Logs/SVM_PCA/svm_pca_rand/$aux_pow/uncompressed_log.txt 2> ../DADOS1/esouza/Logs/SVM_PCA/svm_pca_rand/$aux_pow/uncompressed_errlog.txt &
	wait
done