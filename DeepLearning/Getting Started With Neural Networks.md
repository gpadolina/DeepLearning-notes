# Getting started with neural networks

## Anatomy of a neural network
* *Layers* are combined into a *network* or *model*
* The *input data* and corresponding *targets*
* The *loss function* defines the feedback signal used for learning
* The *optimizer* determines how learning proceeds

![Figure](https://github.com/gpadolina/DeepLearning-notes/blob/master/DeepLearning/Figures/Anatomy%20of%20a%20neural%20network.png)
#### Layers: the building blocks of deep learning
A layer is a data-processing module that takes as input one or more tensors and that outputs one or more tensors. Some layers are stateless, but more frequently layers
have a state: the layer's *weights*, one or several tensors learned with stochastic gradient descent, which together contain the network's *knowledge*.

For instance, simple vector data, stored in 2D tensors of shape (samples, features) is often processed by *densely connected* layers, also called *fully connected* or
*dense* layers (the Dense class in Keras). Sequence data, stored in 3D tensors of shape (samples, timesteps, features) is typically processed by *recurrent* layers such
as an LSTM layer. Image data, stored in 4D tensors, is usually processed by 2D convolution layers (Conv2D).

Building deep-learning models in Keras is done by clipping together compatible layers to form useful data-transformation pipelines. The notion of *layer compatibility*
here refers specifically to the fact that every layer will only accept input tensors of a certain shape and will return output tensors of a certain shape.
