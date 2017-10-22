#!/bin/bash

find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/1/$(echo "{}" | rev | cut -d'/' -f1-1 | rev); python pca.py "{}" $out_filename 1' \; > ../DADOS1/esouza/Logs/PCA/1_log.txt 1> ../DADOS1/esouza/Logs/PCA/1_errlog.txt &
find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/2/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py "{}" $out_filename 2' \; > ../DADOS1/esouza/Logs/PCA/2_log.txt 2> ../DADOS1/esouza/Logs/PCA/2_errlog.txt &
find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/4/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py "{}" $out_filename 4' \; > ../DADOS1/esouza/Logs/PCA/4_log.txt 2> ../DADOS1/esouza/Logs/PCA/4_errlog.txt &
find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/8/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py "{}" $out_filename 8' \; > ../DADOS1/esouza/Logs/PCA/8_log.txt 2> ../DADOS1/esouza/Logs/PCA/8_errlog.txt &
find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/16/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py "{}" $out_filename 16' \; > ../DADOS1/esouza/Logs/PCA/16_log.txt 2> ../DADOS1/esouza/Logs/PCA/16_errlog.txt &
find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/32/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py "{}" $out_filename 32' \; > ../DADOS1/esouza/Logs/PCA/32_log.txt 2> ../DADOS1/esouza/Logs/PCA/32_errlog.txt &
wait
find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/64/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py "{}" $out_filename 64' \; > ../DADOS1/esouza/Logs/PCA/64_log.txt 2> ../DADOS1/esouza/Logs/PCA/64_errlog.txt &
find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/128/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py "{}" $out_filename 128' \; > ../DADOS1/esouza/Logs/PCA/128_log.txt 2> ../DADOS1/esouza/Logs/PCA/128_errlog.txt &
find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/256/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py "{}" $out_filename 256' \; > ../DADOS1/esouza/Logs/PCA/256_log.txt 2> ../DADOS1/esouza/Logs/PCA/256_errlog.txt &
find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/512/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py "{}" $out_filename 512' \; > ../DADOS1/esouza/Logs/PCA/512_log.txt 2> ../DADOS1/esouza/Logs/PCA/512_errlog.txt &
find ../DADOS1/esouza/Datasets/extracted/fc2/ -type f -exec sh -c 'out_filename=../DADOS1/esouza/Datasets/reduced/fc2/1024/$(echo "{}" | rev | cut -d'/' -f1-2 | rev); python pca.py "{}" $out_filename 1024' \; > ../DADOS1/esouza/Logs/PCA/1024_log.txt 2> ../DADOS1/esouza/Logs/PCA/1024_errlog.txt &
wait