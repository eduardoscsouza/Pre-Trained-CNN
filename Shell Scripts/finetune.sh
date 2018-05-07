#!/bin/bash

for size in 8 16 32 64 128 256 512 1024 2048
do
	for qual in 1 5 10 25 50 100
	do
		find ../DADOS1/esouza/Datasets/training/$qual/ -type f -exec bash -c 'dataset=$(echo '"{}"' | rev | cut -d'/' -f1 | cut -d'.' -f2 | rev); python finetuning.py ../DADOS1/esouza/Networks/untrained/'"$size"'/vgg16_$(echo $dataset).h5 '"{}"' ../DADOS1/esouza/Networks/trained/'"$size"'/'"$qual"'/vgg16_$(echo $dataset).h5 ' \; > ../DADOS1/esouza/Logs/Finetuning/$(echo $size)/$(echo $qual)_log.txt 2> ../DADOS1/esouza/Logs/Finetuning/$(echo $size)/$(echo $qual)_errlog.txt;
	done
done