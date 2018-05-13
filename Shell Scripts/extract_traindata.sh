#!/bin/bash

for qual in 1 5 10 25 50 100
do
	find ../DADOS1/esouza/Datasets/compressed/$qual/ -mindepth 1 -maxdepth 1 -type d -exec bash -c 'out_filename=$(echo '"{}"' | rev | cut -d'/' -f1 | rev); python extracttrainingdata.py '"{}"' ../DADOS1/esouza/Datasets/training/'"$qual"'/$(echo $out_filename) 5 ' \; > ../DADOS1/esouza/Logs/Train_Data_Extraction/$qual%_log.txt 2> ../DADOS1/esouza/Logs/Train_Data_Extraction/$qual%_errlog.txt
done
