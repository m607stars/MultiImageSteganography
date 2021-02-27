<h1 align="center"><b>Multi-Image Steganography</b></h1>

This is a [PyTorch](https://pytorch.org) implemenatation of the paper [Multi-Image Steganography Using Deep Neural Networks](https://arxiv.org/pdf/2101.00350.pdf)

# Objective
We will be implementing the [Multi-Image Steganography Using Deep Neural Networks](https://arxiv.org/pdf/2101.00350.pdf), a steaganography technique to encode multiple secret images into a cover image and retreiving the secret images. The authors' original implementation can be found [here](https://github.com/JapsimarSinghWahi/DeepSteganography). We have encoded 3 secret images as per the author's implemenatation, but we can also do it for two images or one single image too.
Here are some examples of object detection in images not seen during training –

# Concepts

* **Steganograhy** - Steganography is a technique of encoding a secret message inside a cover message. In images, traditionally, the message is encoded in the least significant bit (LSB) and the message is retrieved using classical algorthims. 
* **CNNs (Convolutional Neural Network)** - CNNs is a type of neural network which uses a convoltuional filter of given channels and sizes as the weights. CNNs is usually used on images as it can easily extract features of images.
* **Deep Steganography** - CNNs are used to get a representation of the cover(Image used to hide the secret) and the secret images. This encoder generates the cover image that is to be sent to the decoder. This representation is then passed to another CNN which acts as the decoder and generates the secret images. 

# Model Architecture

The model architecture consists of two parts -> Encoder and Decoder 

<p align="center">
<img src="https://github.com/m607stars/MultiImageSteganography/blob/master/assets/model_architecture.png">
</p>

## Encoder

<p align="center">
<img src="https://github.com/m607stars/MultiImageSteganography/blob/master/assets/hidden_image.png">
</p>


Encoder is used to hide the three images into the cover image. It consists of three preperation networks each with a similar architecture.
Each of these prep networks consists of two blocks of the layers. Each of these layer consists of three Conv2d layers with kernel sizes as 3,3,5. These Conv2d layers have output channel dimensions as 50,10 and 5 respectively. Since the output image must have the same dimensions as the input image, the padding is chosen as 1,1,2 respectively and the stride is kept as 1 for all the Conv2d layers. Relu acivation function is applied after each convolution layer. Input image is passed to each layer and the final output from these layers is concatenated and passed on the next block of the prep network. This is carried out for a single image and for a single prep network. Similar procedure is carried out for the remaining two images and we a get three images of dimensions **(batch_size,65,64,64)**.
Note that, here our channels are 65 (Due to concatenation of 5,10 and 50 channel outputs). 

Next, after getting the representation for three images, we need to **hide**  these three images into our cover image. For this, we have a hiding network. First, 
we concat our representations of three images with the cover image, giving as a tensor of dimensions **(batch_size, 198, 64,64)**. Here, the channels are 198, due to concatenation of these four tensors:<br>
* Cover Image (batch_size, 3, 64,64) <br>
* Representation of first secret Image (batch_size, 65, 64,64) <br>
* Representation of second secret Image (batch_size, 65, 64,64) <br>
* Representation of third secret Image (batch_size, 65, 64,64) <br>
<br>

This tensor is passed on the hiding network which consists of five blocks of layers. Each of these layers is similar to the blocks used in the prep network. 
The final layer of the hiding network should output a tensor of dimensions **(batch_size, 3, 64, 64)**. This is the encoded image **OR** the cover images which is used to hide our three secret images. 

## Decoder


<p align="center">
<img src="https://github.com/m607stars/MultiImageSteganography/blob/master/assets/reveal_image.png">
</p>

Decoder is used to reveal the secret image from the encoded cover image. It consists of the three reveal networks for each of the three secret images. Each of the reveal networks consists of five blocks of convolutional layers. The underlying architecture of each block is the same as in the encoder. Since the output of each bock is an image of dimension  **(batch_size,65,64,64)**, we need to change the number of channels to 3 for an RGB image. So, we apply another conv2D layer at the end of each of the reveal networks to get the image of dimesion **(batch_size, 3, 64, 64)**. This is achieved by setting the output channels as 3 in this last conv2D layer. This final image is the decoded secret image obtained from the reveal networks. Thus we get three decoded secret images from each of the three reveal networks.


# Implementation

## Dataset

Dataset used for this paper is [Tiny Image Net](http://cs231n.stanford.edu/tiny-imagenet-200.zip) dataset. This dataset consists of some greyscale images too. These have been removed by running the script [create_dataset.py](https://github.com/m607stars/MultiImageSteganography/blob/master/scripts/create_dataset.py) .
This script is used for creating the training and validation datasets. We sample 10 images from each of the 200 classes of tiny imagenet dataset for training and 4 images for testing. In total, we have **2000** training images, and **800** testing images. 

## Description

* Here, we describe the method to train the model. Initially, this seems like a standard encoder-decoder model. But, the training works in this way: 
First, we train the encoder, and keep the decoder's parameters as untrainable. This helps the encoder to learn and create the encoded hidden image. This model is evaluated using the loss for all the reveal images as well as the hidden image.

* After getting the output from the encoder, we train the decoder (Consisting of three reveal networks). This decoder, is evaluated only on the basis of the reveal loss. 

* Now, the parameters of the deocder's networks are shared with the reveal networks while the encoder is being trained.

* This ideally helps the network to learn much better as the encoder and the decoder network become smaller and task is way more focused in terms of optimization instead of training the whole model in a joint manner. 

## Training

Here are the hyperparameters which we have used for training the networks:

```
IMG_SIZE = 64
LEARNING_RATE  = 0.001
COVER_LOSS_WEIGHT = 1
SECRET_LOSS_WEIGHT = 1
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 1
EPOCHS = 1000
DECODER_LOSS_WEIGHT = 1
```

## Results

Here is a sample output from our testing dataset:

<p align="center">
<img src="https://github.com/m607stars/MultiImageSteganography/blob/master/assets/outputs.png">
</p>

More outputs can be found at the end of our [notebook](https://github.com/m607stars/MultiImageSteganography/blob/master/MultiImageSteganography.ipynb)

## Losses

There are two different losses for the full model and the deocder.

The full model loss is calculated by summation of MSE of the following pairs: <br>
* Cover Image, Hidden Image<br>
* Secret Image 1, Reveal Image 1<br>
* Secret Image 2, Reveal Image 2<br>
* Secret Image 3, Reveal Image 3<br>

We weight the losses of cover image and secret images to 1:1.

For the decoder network, the reveal loss is summation of MSE of the following pairs:<br>

* Secret Image 1, Reveal Image 1<br>
* Secret Image 2, Reveal Image 2<br>
* Secret Image 3, Reveal Image 3<br>

Here is our loss for full model and decoder loss

<p align="center">
<img src="https://github.com/m607stars/MultiImageSteganography/blob/master/assets/full_model_loss.png">
</p>

<p align="center">
<img src="https://github.com/m607stars/MultiImageSteganography/blob/master/assets/decoder_loss.png">
</p>



## Remarks

* We could not use the validation set due to limitations of the GPU. 
* In the author's implementation, for each block, they have used a kernel size of 4 for the second layer. But we have kept it 3.
* We have not added gaussian noise to the encoder's output. 

## Model checkpoint

Model trained on 1000 epochs can be found [here](https://github.com/m607stars/MultiImageSteganography/tree/master/models) 

# Contributors
- [Mayank Chowdhary](https://github.com/m607stars)
- [Talha Chafekar](https://github.com/talha1503)
