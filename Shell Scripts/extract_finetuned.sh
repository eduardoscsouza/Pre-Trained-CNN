#!/bin/bash

for size in 8 16 32 64 128 256 512 1024 2048
do
	for qual in 1 5 10 25 50 100
	do
		network_path=../DADOS1/esouza/Networks/trained/$size/$qual/vgg16_
		out_path=../DADOS1/esouza/Datasets/extracted/fc2/trained/$size/$qual/
		find ../DADOS1/esouza/Datasets/compressed/$qual/ -mindepth 1 -maxdepth 1 -type d -exec bash -c 'out_filename=$(echo '"{}"' | rev | cut -d'/' -f1 | rev); python extractlayeroutput.py '"{}"' $(echo '"$out_path"')$(echo $out_filename).npz fc2 $(echo '"$network_path"')$(echo $out_filename).h5 ' \; > ../DADOS1/esouza/Logs/Finetuning_Extraction/$(echo $size)/$(echo $qual)_log.txt 2> ../DADOS1/esouza/Logs/Finetuning_Extraction/$(echo $size)/$(echo $qual)_errlog.txt
	done
done
