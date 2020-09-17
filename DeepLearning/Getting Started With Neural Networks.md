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

When using Keras, you don't have to worry about compatibility, because the layers you add to your models are dynamically built to match the shape of the incoming layer.
#### Models: networks of layers
The most commong instance is a linear stack of layers, mapping out a single input to a single output.

But as you move forward, you'll be exposed to a much broader variety of network topologies. Someone common ones include the following: two-branch networks, multihead 
networks, and inception blocks.

The topology of a network defines a *hypothesis space*. You may remember that we defined machine learning as "searching for useful representations of some input data,
within a predefined space of possibilities, using guidance from a feedback signal." By choosing a network topology, you constrain your *space of possibilities*
(hypothesis space) to a specific series of tensor operations, mapping input data to output data.
#### Loss functions and optimizers
Once the network architecture is defined, you still have to choose to more things:
* *Loss function (objective function)* - The quantity that will be minimized during training. It represents a measure of success for the task at hand.
* *Optimizer* - Determines how the network will be updated based on the loss function. It implements a specific variant of stochastic gradient descent (SGD).
A neural network that has multiple outputs may have multiple loss functions (one per output). But the gradient-descent process must be based on a *single* scalar loss
value; so for multiloss networks, all losses are combined (via averaging) into a single scalar quantity.

When it comes to common problems such as classification, regression, and sequence prediction, there are simple guidelines you can follow to choose the correct loss. For
instance, you'll use binary crosstentropy for a two-class classification problem, categorical crossentropy for a many-class classification problem, mean-squared error for
a regression problem, connectionist temportal classification (CTC) for a sequence-learning problem, and so on.

## Introduction to Keras
Keras is a deep-learning framework for Python that provides a convenient way to define and train almost any kind of deep-learning model. keras was initially developed for
researches, with the aim of enabling fast experimentation.

Keras has the following key features:
* It allows the same code to run seamlessly on CPU or GPU.
* It has a user-friendly API that makes it easy to quickly prototype deep-learning models.
* It has built-in support convolutional networks (for computer vision), recurrent networks (for sequence processing), and any combination of both.
* It supports arbitrary network architectures: multi-input or multi-output models, layer sharing, model sharing, and so on. This means Keras is appropriate for building
essentially any deep-learning model, from a generative adversarial network to a neural Turing machine.
