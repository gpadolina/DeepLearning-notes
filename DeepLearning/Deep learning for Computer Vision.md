## Deep learning for computer vision
This chapter introduces convolutional neural networks, also known as *convnets*, a type of deep-learning model almost universally used in computer vision applications.
## Introduction to convnents
A convnet takes as input tensors of shape (image_height, image_width, image_channels), not including the batch dimension.
#### The convolution operation
The fundamental difference betwen a densely connected layer and a convolution layer is this: *Dense* layers learn global patterns in their input feature space, whereas
convolution layers learn local patterns: in the case of images, patterns found in small 2D windows of the inputs.

This key characteristics gives convnets two interesting properties:
* *The patterns they learn are translation invariant.* After learning a certain pattern in the lower-right corner of a picture, a convnet can recorgnize it anywhere. A
densely connected network would have to learn the pattern anew if it appeared at a new location. This make convnets data efficient when processing images because the
*visual world is fundamentally translation invariant):* they need fewer training samples to learn representations that have generalization power.
* *They can learn spatial hierarchies of patterns.* A first convolution layer will learn small local patterns such as edges, a second convolutional layer will learn patterns
made of the features of the first layers, and so on. This allows convnets to efficiently learn increasingly complex and abstract visual concepts.

Convolutions operate over 3D tensors, called *feature maps,* with two spatial axes (*height* and *width*) as well as a *depth* axis (also called the *channels* axis). For an
RBG image, the dimension of the depth axis is 3, because the image has three color channels: red, green, and blue. For a black-and-white picture, the depth is 1 levels of gray.

Convolutoins are define by two key parameters:
* *Size of the patches extracted from the inputs* - These are typically 3 x 3 or 5 x 5.
* *Depth of the output feature map* - The number of filters computed by the convolution.

In Keras Conv2D layers, these parameters are the first arguments passed to the layer: Conv2D(output_depth, (window_height, window_width)).

A convolution works by *sliding* these windows of size 3 x 3 or 5 x 5 over the 3D input feature map, stopping at every possible location, and extracting the 3D patch of
surrounding features. Each such 3D patch is then transformed via a tensor product with the same learned weight matrix, called the *convolution kernel* into a 1D vector of
shape.

Note that the output width and height may differ from the input width and height. They may differ for two reasons:
* Border effects, which can be countered by padding the input feature map
* The use of *strides*

If you want to get an output feature map with the same spatial dimensions as the input, you can use *padding.* Padding consists of adding an appropriate number of rows and
columns on each side of the input feature map so as to make it possible to fit center convolution windows around every input tile.

The other factor that can influence output size is the notion of *strides.* The description of convolution so far has assumed that the center tiles of the convolution
windows are all contiguous. But the distance between two successive windows is a parameter of the convolution, called its *stride*, which defaults to 1. It's possible to
have *strided convolutions:* convolutions with a stride higher than 1.

To downsample feature maps, instead of strides, we tend to use the *max-pooling* operation.
#### The max-pooling operation
The role of max pooling is to aggressively downsample feature maps, much like strided convolutions.

In short, the reason to use downsampling is to reduce the number of feature-map coefficients to process, as well as to induce spatial-filter hierarchies by making successive
convolution layers look at increasingly large windows in terms of the fraction of the original input they cover.

Note that the max pooling isn't the only way you can achieve such downsampling. As you already know, you can also use strides in the prior convolution layer. And you can use
average pooling instead of max pooling, where each local input patch is transformed by taking the average value of each channel over the patch, rather than the max. But max
pooling tends to work better than these alternative solutions. In a nutshell, the reason is that features tend to encode the spatial presence of some pattern or concept over
the different tiles of the feature map, and it's more informative to look at the *maximal presence* of different features than at their *average presence.*

## Training a convnet from scratch on a small dataset
As a practical example, we'll focus on classifying images as dogs or cats, in a dataset containing 4,000 pictures of cats and dogs (2,000 each). We'll use 2,000 pictures
for training - 1,000 for validation, and 1,000 for testing.

Then we'll introduce *data augmentation,* a powerful technique for mitigating overfitting in computer vision. By using data augmentation, you'll improve the netwrok to reach
an accuracy of 82%.

In the next section, two more essential techniques for applying deep learning to small datasets will be covered: *feature extraction with a pretrained network* (which will
get to an accuracy of 90% to 96%) and *fine-tuning a pretrained network* (this will get to a final accuracy of 97%).
#### The relevance of deep learning for small-data problems
One fundamental characteristics of deep learning is that it can find interesting features in the training data on its own, without any need for manual feature engineering,
and this can only achieved when lots of training examples are available.

But what constitutes lots of samples is relative - relative to the size and depth of the network you're trying to train, for starters. It isn't possible to train a convnet
to solve a complex problem with just a few tens of samples, but a few hundred can potentially suffice if the model is small and well regularized and the task is simple.
Because convnets learn local, translation-invariant features, they're highly data efficient on perceptual problems. Training a convnet from scratch on a very small image
dataset will still yield reasonable results despite a relative lack of data, without the need for any custom feature engineering.

What's more, deep-learning models are by nature highly repurposable: you can take say an image-classification or speech-to-text model trained on a large-scale dataset and
reuse it on a significantly different problem with only minor changes.
