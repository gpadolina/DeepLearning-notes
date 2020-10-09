## Advanced deep-learning best practices
This explores a number of powerful tools that will bring you closer to being able to develop state-of-the-art models on difficult problems. Using the Keras functional API,
you can build graph-like models, share a layer across different inputs, and use Keras models just like Python functions. Keras callbacks and the TensorBoard browser-based
visualization tool let you monitor models during training.
## Going beyong the Sequential model: the Keras functional API
Until now, all neural networks introduced have been implemented using the *Sequential* model. The *Sequential* model makes the assumption that the network has exactly one
input and exactly one output, and that it consists of a linear stack of layers.

This set of assumptions is too inflexible in a number of cases. Some networks require several independent inputs, others require multiple outputs, and some networks have
internal branching between layers that makes them look like *graphs* of layer rather than linear stacks of layers.

Some tasks, for instance, require *multimodal* inputs: they merge data coming from different input sources, processing each type of data using different kinds of neural
layers. Imagine a deep-learning model trying to predict the most likely market price of a second-hand piece of clothing using the following inputs: user-provided metadata,
a user-provided text description, and a picture of item. A naive approach would be to train three separate models and then do a weighted average of their predictions. But
this may be suboptimal, because the information extracted by the models may be redundant. A better way is to *jointly* learn a more accurate model of the data by using a
model that can see all available input modalities simultaneously: a model with three input branches.

Additionally, many recently developed neural architectures require nonlinear network topology: networks structured as directed acyclic graphis. The Inception family of
networks for instance, relies on *Inception modules,* where the input is processed by several parallel convolutoinal branches whose outputs are then merge back into a single
tensor. There's also the recent trend of adding *residual connections* to a model, which started with the ResNet family of networks. A residual connection consists of
reinjecting previous representations into the downstream flow of data by adding a past output tensor to a later output tensor, which helps prevent information loss along
the data data-processing flow.

These three important use cases - multi-input models, multi-output models, and graph-like models - aren't possible when using only the *Sequential* model class in Keras.
But there's another far more general and flexible way to use Keras: the *functional API.*
#### Introduction to the functional API
In the functional API, you directly manipulate tensors, and you use layers as *functions* that take tensors and return tensors.

Behind the scenes, Keras retrieves every layer involved in going from *input_tensor* to *output_tensor,* bringing them together into a graph-like data structure - a *Model.*

When it comes to compiling, training, or evaluating such an instance of *Model,* the API is the same as that of *Sequential.*
#### Multi-input models
The functional API can be used to build models that have multiple inputs. Typically, such models at some point merge their different input branches using a layer that can
combine several tensors: by adding them, concatenating them, and so on. This is usually done via a Keras merge operation such as *keras.layers.add, keras.layers.concatenate,*
and so on.

How do you train two-input model? There are two possible APIs: you can feed the model a list of Numpy arrays as inputs, or you can feed it a dictionary that maps input names
to Numpy arrays. Naturally, the latter option is available only if you give names to your inputs.
#### Multi-output models
Importantly, training such a model requires the ability to specify different loss functions for different heads of the network: for instance, age prediction is a scalar
regression task, but gender prediction is a binary classification task, requiring a different training procedure. But because gradient descent requires you to minimize a
*scalar,* you must combine these losses into a single value in order to train the model. The simplest way to combine different losses is to sum them all. In Keras, you can
use either a list or a dictionary of losses in *compile* to specify different objects for different outputs; the resulting loss values are summed into a global loss, which
is minimized during training.

Note that every imbalanced loss contributions will cause the model representations to be optimized preferentially for the task with the largest individual loss, at the
expense of the other tasks. To remedy this, you can assign different levels of importance to the loss values in their contribution to the final loss. This is useful in
particular if the losses' values use different scales.
#### Directed acyclic graphs of layers
With the functional API, not only you can build models with multiple inputs and multiple outputs, but you can also implement networks with a complex internal topology.
Neural networks in Keras are allowed to be arbitrary *directed acyclic graphs* of layers. The qualifier *acyclic* is important: these graphs can't have cycles. It's impossible
for a tensor x to become the input of one of the layers that generated x. The only processing *loops* that are allowed are those internal to recurrent layers.

Several common neural-network components are implemented as graphs. Two notable ones are Inception modules and residual connections.
#### Inception modules
*Inception* is a popular type of network architecture for convolutional neural networks inspired by the earlier *network-in-network* architecture. It consists of a stack of
modules that themselves look like small independent networks, split into several parallel branches.

Another closely related model available as part of the Keras applications module is *Xception.* Xception, which stands for *extreme inception,* is a convnet architecture
loosely inspired by Inception.It takes the idea of separating the learning of channel-wise and space-wise features ot its logical extreme, and replaces Inception modules
with depth-wise separable convolutions consisting of a depthwise convolution (a spatial convolution where every input channel is handled separately) followed by a pointwise
convolution (a 1x1 convolution) - effectively, an extreme form of an Inception module, where spatial features and channel-wise features are fully separated.
#### Residual connections
*Residual connections* are a common graph-like network component found in many post-2015 network architectures, including Xception. They tackle two common problems that
plague any large-scale deep-learning model: vanishing gradients and representational bottlenecks. In general, adding residual connections to any model that has more than
10 layers is likely to be beneficial.
#### Layer weight sharing
One more important feature of the functional API is the ability to reuse a layer instance several times. When you call a layer instance twice, instead of instantiating a
new layer for each call, you reuse the same weights with every call. This allows you to build models that have shared branches - several branches that all share the same
knowledge and perform the same operations.

In this setup, the two input sentences are interchangeable, because semantic similarity is a symmetrical relationship: the similarity of A to B is identical to the similarity
of B to A. For this reason, it wouldn't make sense to learn two independent models for processing each input sentence. Rather, you want to process both with a single *LSTM*
layer. The representations of this *LSTM* layer (its weights) are learned based on both inputs simultaneously. This is what we call a *Siamese LSTM* model or a *shared LSTM.*

Naturally, a layer instance may be used more than once - it can be called arbitrarily many times, reusing the same set of weights every time.
## Inspecting and monitoring deep-learning models using Keras callbacks and TensorBoard
Launching a training run on a large dataset for tens of epochs using *model.fit( )* or *model.fit_generator( )* can be a bit like launching a paper airplane: past the initial
impulse, you don't have any control over its tracjectory or its landing spot. The techniques presented here will transform the call to *model.fit( )* from a paper airplane
into a smart, autonomous drone that can self instrospect and dynamically take action.
#### Using callbacks to act on a model during training
A much better way to handle this is to stop training when you measure tha tthe validation loss is no longer improving. This can be achieved using a Keras callback. A
*callback* is an object (a class instance implementing specific methods) that is passed to the model in the call to *fit* and that is called by the model at various points
during training. It has access to all the available data about the state of the model and its performance, and it can take action: interrupt training, save a model, load a
different weight set, or otherwise alter the state of the model.

Here are some examples of ways you can use callbacks:
* *Model checkpointing* - Saving the current weights of the model at different points during training.
* *Early stopping* - Interrupting training when the validation loss is no longer improving and saving the best model obtained during training.
* *Dynamically adjusting the value of certain parameters during training* - Such as the learning rate of the optimizer.
* *Logging training and validation metrics during training, or visualizing the representations learned by the models as they're updated* - The Keras progress bar that
you're familiar with is a callback!
#### The modelcheckpoint and earlystopping callbacks
You can use the *EarlyStopping* callback to interrupt training once a target metric being monitored has stopped improving for a fixed number of epochs. For instance, this
callback allows you to interrupt training as soonas you start overfitting, thus avoiding having to retrain your model for a smaller number of epochs. This callback is
typically used in combination with *ModelCheckpoint,* which lets you continually save the model during training (and, optionally, save only the current best model so far:
the version of the model that achieved the best performance at the end of an epoch).
#### The reduceLRonplateau callback
You can use this callback to reduce the learning rate when the validation loss has stopped improving. Reducing or increasing the learning rate in case of a *loss plateau* is
an effective strategy to get out of local minima during training.
#### Introduction to TensorBoard: the TensorFlow visualization framework
To do good research or develop good models, you need rich, frequent feedback about what's going on inside your models during your experiments. That's the point of running
experiments: to get information about how well a model performs - as much as information as possible. Making progress is an iterative process or loop: you start with an
idea and express it as an experiment, attempting to validate or invalidate your idea.
