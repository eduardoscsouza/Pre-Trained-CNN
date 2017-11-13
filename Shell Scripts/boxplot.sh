#!/bin/bash

for type in imgnet rand
do
	filter=*$(echo $type)*.npz
	for dataset in corel-1000 tropical_fruits1400 coil-20 17flowers CUB_200_2011
	do
		dataname=$(echo $dataset)_$(echo $type).npz
		for comp in 4096 1024 512 256 128 64 32 16 8
		do
			echo $dataname
			echo $comp
			echo $filter
			echo ../DADOS1/esouza/Results/Boxplot/$(echo $type)/$(echo $dataset)_$(echo $comp).pdf
			python boxplot.py ../DADOS1/esouza/Results/ $dataname $comp $filter ../DADOS1/esouza/Results/Boxplot/$(echo $type)/$(echo $dataset)_$(echo $comp).pdf
		done
	done
done