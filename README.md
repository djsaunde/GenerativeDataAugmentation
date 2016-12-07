# Generative Data Augmentation

This repository contains code for our final project for CS 697L - Deep Learning. 

The goal of this project is to use a variant of GANs in order to learn transformations of training data which will be used to augment the training dataset. We call
the generator a "transformer" instead, since this network plays a slightly different functional role than a generator network might.

The transformer will take a minibatch of training data, perform a nonlinear transformation on each datum, and will try to maximize the probability that the discriminator 
network classifies it incorrectly, as being from the training dataset, while minimizing the correlation between the transformed image and the original training
data point. The discriminator will take inputs from both the training dataset and the transformer network, and will try to maximize the probability that
it correctly classifies its input as either from the training dataset (and therefore, its label), or from the transformer (and perhaps try to infer the class label
with which the original image is associated).

An initial approach will be to build the generator and discriminator as in [1], and then apply the architectures to the CIFAR-10 dataset
as in [2]. We can try to use the generated images to augment our training dataset, and benchmark results compared to a baseline architecture
(say, as pretrained ConvNet for CIFAR-10 classification, vs. using retraining this network as a discriminator).

From here, we will try to modify to the transformer architecture described above to transform an training data image in a useful way, rather than
attempting to generate images from the training dataset distribution from random noise. We must develop a way to ensure that
the original image and the transformed image are not highly correlated.

# References

- [1] Generative Adversarial Networks (https://arxiv.org/pdf/1406.2661v1.pdf)

- [2] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (https://arxiv.org/abs/1511.06434)

# Related Repositories

- Generative Adversarial Networks (https://github.com/goodfeli/adversarial)

- Deep Convolutional Generative Adversarial Networks (https://github.com/Newmu/dcgan_code)
