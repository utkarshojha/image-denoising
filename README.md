# Introduction

This repository contains the code for our work on densoising high resolution images using deep learning [paper](http://ieeexplore.ieee.org/document/7838260/). Present state-of-the-art methods like BM3D, KSVD and Non-local means do produce high quality denoised results. But when the size of image becomes very high, for ex. 4000 x 80000 pixels, those high quality results come at a cost of high computational time. This time consuming factor serves as a motivation to come up with a model that can provide comparable results, if not better, in much less time. So, I've used a deep learning approach that automatically tries to learn the function that maps a noisy image to its denoised version. I've used thenao as the deep learning framework, and have worked on the [publicly available codes](https://github.com/lisa-lab/DeepLearningTutorials) provided by the [MILA Lab](https://mila.umontreal.ca/). 

# Data
Unfortunately, the complete data on which I actually trained this model cannot be released publicly, since I used data that belong to ISRO (images captured by CARTOSAT 2), although I have included snippets of one or two images in the results section, to provide a sense of what the data looks like. But anyone can easily use their own data (black and white noisy and denoised images for now) and train the model accordingly.

# Dependency
```
python 2.7, theano
```
# Results

## Graphs
### Variation of training process with different patch size
![](/graphs/patches_plot.png)
### Variation of training process with different number of hidden layers
![](/graphs/variation_with_no_of_hidden_layers1.png)
### Variation of training process with different sizes of hidden layers
![](/graphs/variation_with_size_of_hidden_layers.png)











