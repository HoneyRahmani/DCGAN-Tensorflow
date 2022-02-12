
## Table of Content
> * [Deep Convolutional Generative Adversarial Networks (DCGANs) - Tensorflow](#DeepConvolutionalGenerativeAdversarialNetworks(GANs)-Tensorflow)
>   * [About the Project](#AbouttheProject)
>   * [About Database](#AboutDatabases)
>   * [Built with](#Builtwith)
>   * [Installation](#Installation)

# Deep Convolutional Generative Adversarial Networks (DCGANs) - Tensorflow
## About the Project
This project focuses on develop the Deep Convolutional GAN (DCGAN) to generate new images similar to the STL-10 dataset using PyTorch.

![Gan](https://user-images.githubusercontent.com/75105778/153688204-0a4fdaae-d7c0-44b8-b3c2-e95b0185e04d.jpg)

The generator generates fake data and the discriminator identifies real images from fake images. The generator and the discriminator compete with each other in a game in the training stage. This competition is generating better-looking images to deceive the discriminator by the generator and getting better at identifying real images from fake images by the discriminator.


## About Database

Dataset is fashion_mnist dataset from the keras.

## Built with
* Pytorch
* Binary cross-entropy (BCE) loss function for both of generator and discriminator.
* Adam optimizer


