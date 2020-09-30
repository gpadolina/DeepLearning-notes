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
