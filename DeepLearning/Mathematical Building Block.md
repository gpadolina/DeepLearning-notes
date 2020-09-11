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
### Video data
Video data is one of the few types of real-world data for which you'll need 5D tensors. A video can be understood as a sequence of frames, each frame being a color image.
Because each frame can be stored in a 3D tensor (height, width, color_depth), a sequence of frames can be stored in a 4D tensor (frames, height, width, color_depth),
and thus a batch of different videos can be stored in a 5D tensor of shape (samples, frames, height, width, color_depth).
