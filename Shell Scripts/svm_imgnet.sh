#!/bin/bash

for per in {10..90..20}
do
	find ../DADOS1/esouza/Datasets/extracted/fc2/$per%/ -type f -name "*imgnet.npz" -exec bash -c 'python svm.py "{}" ../DADOS1/esouza/Results/SVM/"$per"%_imgnet.txt ../DADOS1/esouza/Results/SVM/"$per"%_imgnet.npz' \; > ../DADOS1/esouza/Logs/SVM/svm_imgnet/$per%_log.txt 2> ../DADOS1/esouza/Logs/SVM/svm_imgnet/$per%_errlog.txt &
done
find ../DADOS1/esouza/Datasets/extracted/fc2/uncompressed/ -type f -name "*imgnet.npz" -exec bash -c 'python svm.py "{}" ../DADOS1/esouza/Results/SVM/uncompressed_imgnet.txt ../DADOS1/esouza/Results/SVM/uncompressed_imgnet.npz' \; > ../DADOS1/esouza/Logs/SVM/svm_imgnet/uncompressed_log.txt 2> ../DADOS1/esouza/Logs/SVM/svm_imgnet/uncompressed_errlog.txt &
wait