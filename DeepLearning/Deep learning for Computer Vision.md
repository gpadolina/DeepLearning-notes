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
