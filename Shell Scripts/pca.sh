#!/bin/bash

for pow in {3..10}
do
	aux_pow=$((2**$pow))
	find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec bash -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/'"$aux_pow"'/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py '"{}"' $out_filename '"$aux_pow"' ' \; > ../DADOS1/esouza/Logs/PCA/$aux_pow_log.txt 2> ../DADOS1/esouza/Logs/PCA/$aux_pow_errlog.txt &
done
wait