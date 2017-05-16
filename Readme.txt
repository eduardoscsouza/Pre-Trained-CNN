python downloadvgg16.py <in_height> <in_width> <in_channels> <out_classes> <filename>
Downloads the convolutional layers of the VGG16 network trained on the ImageNet dataset, and
adds an input layer for images with (in_height, in_width, in_channels) format, the fully connected
untrained layers and an output layer for out_classes possible classes
Ex:
	python downloadvgg16.py 32 32 3 10 ../Data/vgg16_imnet.h5
	Network for 32x32x3 images with 10 possible classes to be saved in the Data directory with name vgg16_imnet.h5
The default value for the argumenst are:
in_height	=	224
in_width	=	224
in_channels	=	3
out_classes	=	2
filename	=	vgg16_imgnet.h5