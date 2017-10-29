#!/bin/bash

for qual in 1 5 10 25 50
do
	find ../DADOS1/esouza/Datasets/classified -mindepth 1 -maxdepth 1 -type d -exec bash -c 'dirname=$(echo '"{}"' | rev | cut -d'/' -f1 | rev); mkdir ../DADOS1/esouza/Datasets/compressed/'"$qual"'/$dirname' \;
	find ../DADOS1/esouza/Datasets/classified -mindepth 2 -maxdepth 2 -type d -exec bash -c 'dirname=$(echo '"{}"' | rev | cut -d'/' -f1-2 | rev); mkdir ../DADOS1/esouza/Datasets/compressed/'"$qual"'/$dirname' \;
done

for qual in 1 5 10 25 50
do
	find ../DADOS1/esouza/Datasets/classified -type f -exec bash -c 'out_filename=$(echo '"{}"' | rev | cut -d'/' -f1-3 | cut -d'.' -f2- | rev ).jpg; convert '"{}"' -quality '"$qual"' ../DADOS1/esouza/Datasets/compressed/'"$qual"'/$out_filename; echo $filename_to_../DADOS1/esouza/Datasets/compressed/'"$qual"'/$out_filename_with_'"$qual"'_quality' \; > ../DADOS1/esouza/Logs/Compression/$qual%_log.txt 2> ../DADOS1/esouza/Logs/Compression/$qual%_errlog.txt &
done
cp -rf ../DADOS1/esouza/Datasets/classified/* ../DADOS1/esouza/Datasets/compressed/100/ > ../DADOS1/esouza/Logs/Compression/100%_log.txt 2> ../DADOS1/esouza/Logs/Compression/100%_errlog.txt &
wait