#!/bin/bash

for size in 8 16 32 64 128 256 512 1024 2048
do
	find ../DADOS1/esouza/Datasets/training/100/ -type f -exec bash -c 'dataset=$(echo '"{}"' | rev | cut -d'/' -f1 | cut -d'.' -f2 | rev); python generatecnn.py ../DADOS1/esouza/Networks/vgg16_imgnet.h5 '"$size"' '"{}"' ../DADOS1/esouza/Networks/untrained/'"$size"'/vgg16_$(echo $dataset).h5 ' \; > ../DADOS1/esouza/Logs/Generate_CNNS/$(echo $size)_log.txt 2> ../DADOS1/esouza/Logs/Generate_CNNS/$(echo $size)_errlog.txt
done