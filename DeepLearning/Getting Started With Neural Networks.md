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
#### Developing with Keras
The typical workflow looks just like that example:
1. Define your training data: input tensors and target tensors.
2. Define a network of layers or model that maps your inputs to your targets.
3. Configure the learning process by choosing a loss function, an optimizer, and some metrics to monitor.
4. Iterate on your training data by calling the *fit( )* methof of your model.

There are two ways to define a model: using the *Sequential* class (only for linear stacks of layers, which is the most common network architecture by far) for the
*functional API* (fore directed acyclic graphs of layers, which lets you build completely arbitary architectures).
## Setting up a deep-learning workstation
#### Jupyter notebooks: the preferred way
Jupyter notebooks are a great way to run deep-learning experiments. A notebook allows you to break up long experiments into smaller pieces that can be executed 
independently, which makes development interactive and means you don't have to rerun all of your previous code if something goes wrong late in an experiment.
#### Best GPU for deep learning
The first thing to note is that it must be an NVIDIA GPU. NVIDIA is the only graphics computing company that has invested heavily in deep learning so far and modern
deep-learning frameworks can only run on NVIDIA cards.
## Classifying movie reviews: a binary classification example
Two-class classification or binary classification may be the most widely applied kind of machine-learning problem. In this example, you'll learn to classify movie
review as positive or negative, based on the text content of the reviews.
#### The IMDB dataset
The IMDB dataset: a set of 50,000 highly polarized reviews from the Internet Movie Database. They're split into 25,000 reviews for training and 25,000 reviews for testing,
each set consisting of 50% of negative and 50% positive reviews.
```
# Loading the IMDB dataset
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

train_data[0]
train_labels[0]

max([max(sequences) for sequence in train_data])

word_index = imdb.get_word_index      # word_index is a dictionary mapping words to an integer index
reverse_word_index = dict(
  [(value, key) for (key, value) in word_index.items()]) # Reverses it, mapping integer indices to words
decoded_review = ' '.join(
  [reverse_word_index.get(i - 3, '?') for in in train_data[0]]) # Decodes the review. Note that the indices are offset by 3
                                      # because 0, 1, and 2 are reserved indices for "padding", "start of sequence", and "unknown"
```
#### Preparing the data
You can't feed lists of integers into a neural network. You have to turn your lists into tensors. There are two ways to do that:
* Pad your lists so that they all have the same length, turn them into an integer tensor of shape (samples, word_indices), and then use as the first layer in your network
a  layer of capable of handling such integer tensors
* One-hot encode your lists to turn them into vectors of 0s and 1s
```
# Encoding the integer sequences into a binary matrix
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))       # creates an all-zero matrix of shape (len(sequences), dimension)
  for i, sequence in enumerate(sequences):
      results[i, sequence] = 1.                         # sets specific indices of results[i] to 1s
  return results
  
x_train = vectorize_sequences(train_data)               # vectorized training data
x_test = vectorize_sequences(test_data)                 # vectorized test data
```
#### Building your network
The input data is vectors and the labels are scalars (1s and 0s). A *hidden unit* is a dimension in the representation space of the layer.

There are two key architecture to be made about such a stack of Dense layers.
* How many layers to use
* How many hidden units to choose for each layer

A relu (rectified linear unit) is a function meant to zero out negative values, whereas a sigmoid squashes arbitrary values into the [0,1] interval, outputting
something that can be interpreted as a probability.
```
# Model definition
from keras import models
from keras import layers

model = models.Sequential( )
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
mdoel.add(layers.Dense(1, activation='sigmoid'))
```
Without an activation function like relu, the Dense layer would consist of two linear operations - a dot product and an addition.

Finally, choose a loss function and an optimizer. Because you're facing a binary classification problem and the output of your network is a probability, it's best to
use the binary_crossentropy loss. Crossentropy is usually the best choise when you're dealing with models that output probabilities. *Crossentropy* is a quantity from
the field of Information Theory that measures the distance between probability distributions or, in this case, between the ground-truth distribution and your
predictions.
```
# Compiling the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```
You're passing your optimizer, loss function, and metrics as strings, which is possible because rmsprop, binary_crossentropy, and accuracy are packaged as part of
Keras.
```
# Configuring the optimizer
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
```

```
# Uisng custom losses and metrics
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
            loss=losses.binary_crossentropy,
            metrics=[metrics.binary_accuracy])
```
#### Validating your approach
```
# Setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
```

```
# Training your model
model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['acc']
            
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
```
On CPU, this will take less than 2 seconds per epoch - training is over in 20 seconds. At the end of every epoch, there is a slight pause as the model computes its
loss and accuracy on the 10,000 samples of the validation data.
```
# Plotting the training and validation loss
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')          # 'bo' is for blue dot
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')     # 'b' is for solid blue line
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend( )
```

```
# Plotting the training and validation accuracy
plt.clf( )  # clears the figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend( )
```

```
# Retraining a model from scratch
model = models.Sequential( )
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])
            
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
```
