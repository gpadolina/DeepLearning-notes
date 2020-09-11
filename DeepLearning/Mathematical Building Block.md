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
