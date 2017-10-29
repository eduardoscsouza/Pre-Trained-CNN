#!/bin/bash

for qual in 1 5 10 25 50 100
do
	find ../DADOS1/esouza/Datasets/compressed/$qual/ -mindepth 1 -maxdepth 1 -type d -exec sh -c 'out_filename=$(echo "{}" | rev | cut -d'/' -f1 | rev); python extractlayeroutput.py '"{}"' ../DADOS1/esouza/Datasets/extracted/fc2/'"$qual"'/$(echo $out_filename)_imgnet fc2 ../DADOS1/esouza/Networks/vgg16_imgnet.h5' \; > ../DADOS1/esouza/Logs/Extraction/extract_imgnet/$qual%_log.txt 2> ../DADOS1/esouza/Logs/Extraction/extract_imgnet/$qual%_errlog.txt
done