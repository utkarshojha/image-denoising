# Introduction
This project tackles the problem of denoising high resolution multispectral images using deep learning approach. I've used stacked autoencoders as the deep learning model to automatically learn the function that maps noisy image to its denoised version. I've worked on code provided by the deep learning tutorial by MILA Lab using theano.

# Architecture of the model
For the first three spectral bands (blue, green and red), the architecture of the model comes out to be same with layers configuration given as [676,2620,2620,2620,2620,676]. For the last band (near infrared), the architecture is [676,2480,2480,2480,2480,676]. The input layer takes a noisy patch size of 26 x 26, and outputs a denoised patch of size 26 x 26. 
The activation function used for every neuron of every layer is sigmoid function (one can play with different kinds of activation function like tanh, Relu etc.).




