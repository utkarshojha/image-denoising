# Introduction
This project tackles the problem of denoising high resolution multispectral images using deep learning approach. Present state-of-the-art methods like BM3D, KSVD and Non-local means do produce high quality denoised results. But when the size of image becomes very high, for ex. 4000 x 80000 pixels, those high quality results come at a cost of high computational time. This time consuming factor serves as a motivation to come up with a model that can provide comparable results, if not better, in much less time. So, I've used a deep learning approach that automatically tries to learn the function that maps a noisy image to its denoised version. I've used thenao as the deep learning framework, and have worked on the publicly available codes provided by the MILA Lab. 

#Data
Unfortunately, the complete data on which I actually trained this model cannot be released publicly, since I used Indian Space Research Organisation's satellite captured images, although I have included snippets of one or two images, to provide a sense of what the data looks like. But anyone can easily use their own data i.e black and white noisy and denoised images and train the model accordingly.

#Dependency
This uses python + theano + numpy as its major dependencies. Also, if you want to train the model on a different dataset, I'd recommend doing it on a GPU cluster rather than simply on a laptop or desktop. It took 2-3 days for the model to train on a powerful GPU, so I guess one can imagine how much time would it require for a CPU to perform the same operation. 









