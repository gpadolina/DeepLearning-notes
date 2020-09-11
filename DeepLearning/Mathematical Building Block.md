# Mathematical building blocks of neural networks

The cord building block of neural networks is the *layer*, a data-preprocessing module that you can think of as a filter for data. Some data goes in and it comes out
in a more useful form. Specifically, layers extract *representations* out of the data fed into them - hopefully, representation that are more meaningful for the problem
at hand.

To make a network ready for training, we need to pick three more things, as part of the *compilation* step:
* *Loss function* - How the network will be able to measure its performance on the training data, and thus how it will be able to steer itself in the right direction.
* *Optimizer* - The mechanism through which the network will update itself based on the data it sees and its loss function.
* *Metrics to monitor during training and testing*
The gap between training accuracy and test accuracy is an example of *overfitting*: the fact that machine-learning models tend to perform worse on new data than on their
training data.

## Data representation for neural networks
In general, all current machine-learning systems use *tensors*, multidimensional Numpy arrays, as their basic data structure. Tensors are fundamental to the field. At
its core, a tensor is a container for data - almost always numerical data. Tensors are a generalization of matrices to an arbitrary number of dimensions. A *dimension*
is often called an *axis* in the context of tensors.
#### Scalars (0D tensors)
A tensor that contains only one number is called a *scalar* (or scalar tensor, 0-dimensional tensor, or 0D tensor). The number of axes of a tensor is also called its
*rank*.
#### Vectors (1D tensors)
An array of numbers is called a *vector*, or 1D tensor. A 1D tensor is said to have exactly one axis. *Dimensionality* can denote either the number of entries along
specific axis or the number of axes in a tensor.
#### Matrices (2D tensors)
An array of vectors ia *matrix* or 2D tensor. A matrix has two axes (*rows* and *columns*).
#### 3D tensors and higher-dimensional tensors
If you pack such matrices in a new array, you obtain a 3D tensor, which you can visually interpret as a cube of numbers. By packing 3D tensors in an array, you can
create a 4D tensor and so on. In deep learning, you'll generally manipulate tensors that are 0D to 4D, although you may go up to 5D if you process video data.
#### Key attributes
A tensor is defined by three key attributes:
* *Number of axes (rank)* - For instance, a 3D tensor has three axes and a matrix has two axes.
* *Shape* - This is a tuple of integers that describes how many dimensions the tensor has along each axis.
* *Data type* - This is the type of the data contained in the tensor.
#### Manipulating tensors in Numpy
Selecting specific elements in a tensor is called *tensor slicing*.
#### Data batches
When considering such a batch tensor, the first axis (axis 0) is called the *batch axis* or *batch dimension*. In addition, deep-learning models don't process an entire
dataset at once. They break the data into small batches.
#### Real-world examples
The data you'll manipulate will almost always fall into one of the following categories:
* *Vector data* - 2D tensors of shape (samples, features)
* *Timeseries data or sequence data* - 3D tensors of shape (samples, timesteps, features)
* *Images* - 4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
* *Video* - 5D tensors of shape (samples, frames, height, width, channels) or (samples, frames, channels, height, width)
#### Vector data
This is the most common case. In such a dataset, each single data point can be encoded as a vector, and thus a batch of data will be encoded as a 2D tensor, where the
first axis is the *samples axis* and the second axis is the *features axis*.
#### Timeseries data or a sequence data
Whenever time matters in your data, it makes sense to store it in a 3D tensor with an explicit time axis. The time axis is always the second axis (axis of index 1),
by convention.
#### Image data
Images typicaly have three dimensions: height, width, and color depth. There are two conventions for shapes of images tensors: the *channels-last* convention (Tensorflow)
and the *channels-first* convention (Theano). The TensorFlow machine-learning framework places the color-depth axis at the end: (samples, height, width, color_depth).
Meanwhile, Theano places the color depth axis right after the batch axis: (samples, color_depth, height, width).
#### Video data
Video data is one of the few types of real-world data for which you'll need 5D tensors. A video can be understood as a sequence of frames, each frame being a color image.
Because each frame can be stored in a 3D tensor (height, width, color_depth), a sequence of frames can be stored in a 4D tensor (frames, height, width, color_depth),
and thus a batch of different videos can be stored in a 5D tensor of shape (samples, frames, height, width, color_depth).

## Tensor operations
Much as any computer program can be ultimately reduced to a small set of binary operations on binary inputs (AND, OR, NOR, and so on), all transformation learned by
deep neural networks can be reduced to a handful of *tensor operations* applied to tensors of numeric data.
#### Element-wise operations
The relu operation and addition are *element-wise* operations: operations that are applied independently to each entry in the tensors beind considered. This means these
operations are highly amenable to massively parallele implementations.
#### Broadcasting
When possiblle, the smaller tensor will be *broadcaster* to match the shape of the larger tensor. Broadcasting consists of two steps:
* Axes (called *broadcast axes*) are added to the smaller tensor to match the ndim of the larger tensor.
* The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor.
#### Tensor dot
The dot operation, also called a *tensor product* is the most common, most useful tensor operations. Contrary to element-wise operations, it combines entries in the
input tensors. An element-wise product is done with the asterisk operator in Numpy, Keras, Theano, and Tensorflow.dot uses a different syntax in TensorFlow.

You can take the dot product of the two matrices x and y (dot(x, y)) if and only if x.shape[1] == y.shape[0]. The result is a matrix with shape (x.shape[0], y.shape[1]).
#### Tensor reshaping
A third type of tensor operation that's essential to understand is *tensor reshaping*. Reshaping a tensor means rearranging its rows and columns to match a target shape.
Naturally, the reshaped tensor has the same total number of coefficients as the initial tensor.

A special case of reshaping that's commonly encountered is *transposition*. *Transposing* a matrix means exchanging its rows and its columns so that x[i, :] becomes
x[:, i].
#### A geometric interpretation of deep learning
Uncrumpling paper balls is what machine learning is about: finding neat representations for complex, highly folded data manifolds. Each layer in a deep network applies
a transformation that disentangles the data a little - and a deep stack of layers makes tractable an extremely complicated disentanglement process.