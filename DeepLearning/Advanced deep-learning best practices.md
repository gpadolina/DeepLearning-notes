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

Keras helps you go from idea to experiment in the least possible time, and fast GPUs can help you get from experiment to result as quickly as possible. But what about
processing the experiment results? That's where TensorBoard comes in.

This section introduces TensorBoard, a browser-based visualization tool that comes packaged with TensorFlow. Note that it's only available for Keras models when you're using
Keras with the TensorFlow backend. TensorBoard gives you access to several neat features, all in your browser:
* Visually monitoring metrics during training
* Visualizing your model architecture
* Visualizing histograms of activations and gradients
* Exploring embeddings in 3D
Before you start using TensorBoard, you need to create a directory where you'll store the log files it generates.

At this point, you can launch the TensorBoard server from the command line, instructing it to read the logs the callback is currently writing. The *tensorboard* utility
should have been automatically installed on your machine the moment you installed TensorFlow.

In addition to live graphs of the trianing and validation metrics, you get access to the Histograms tab, where you can find pretty visualizations of histograms of activation
values taken by your layers.

The Graphs tab shows an interactive visualization of the graph of low-level TensorFlow operations underlying your Keras model.

Note that Keras also provides another, cleaner way to plot models as graphs of layers rather than graphs of TensorFlow operations: the utility *keras.utils.plot_model.*
Using it requires that you've installed the Python *pydot* and *pydot-ng* libraries as well as the *graphviz* library.
#### Wrapping up
* Keras callbacks provide a simple way to monitor models during training and automatically take action based on the state of the model.
* When you're using TensorFlow, TensorBoard is a great way to visualize model activity in your browser. You can use it in Keras models via the *TensorBoard* callback.
## Getting the most out of your models
In this section, we'll go beyond "works okay" to "works great and wins machine-learning competitions" by offering you a quick guide to a set of must-know techniques for
building state-of-the-art deep-learning models.
#### Advanced architecture patterns
We covered one important design pattern in detail in the previous section: residual conenctions. There are two more design patterns you should know about: normalization
and depthwise separable convolution.
#### Batch normalization
*Normalization* is a broad category of methods that seek to make different samples seen by a machine-learning model more similar to each other, which helps the model learn
and generalize well to new data. The most common form of data normalization is one you've seen several times in this book already: centering the data on 0 by subtracting
the mean from the data, and giving the data a unit standard deviation by dividing the data by its standard deviation.

Batch normalization is a type of layer (*BatchNormalization* in Keras) introduced in 2015; it can adaptively normalize data even as the mean and variance change over time
during training. For instance, *BatchNormalization* is used liberally in many of the advanced convnet architectures that come packaged with Keras, such as ResNet50,
Inception, V3, and Xception.

The *BatchNormalization* layer is typically used after a convolutional or densely connected layer.

The *BatchNormalization* layer takes an axis argument, which specifies the feature axis that should be normalized. This argument defaults to -1, the last axis in the input
tensor. This is the correct value when using *Dense* layers, *Conv1D* layers, RNN layers, and *Conv2D* layers with *data_format* set to *"channels_last".* But in the niche
use case of *Conv2D* layers with *data_format* set to *"channels_first"*, the features axis is axis 1; the *axis* argument in *BatchNormalization* should accordingly be
set to 1.

#### Depthwise separable convolution
There's a layer you can use as a drop-in replacement for *Conv2D* that will make your model lighter (fewer trainable weight parameters) and faster (fewer floating-point
operations) and cause it to perform a few percentage points better on its task. That is precisely what the *depthwise separable convolution* layer does (*SeparableConv2D*).
This layer performs a spatial convolution on each channel of its input, independently, before mixing output channels via a pointwise convolution.

These advantages become especially important when you're training small models from scratch on limited data. For instance, here's how you can build a lightweight, depthwise
separable convnet for an image-classification task (softmax categorical classification).

When it comes to large-scale models, depthwise separable convolutions are the basis of Xception architecture, a high-performing convnet that comes packaged with Keras.
#### Hyperparameter optimization
When building a deep-learning model, you have to make many seemingly arbitrary decisions: How many layers should you stack? How many units or filters should go in each layer?
Should you use *relu* as activation, or a different function? Should you use *BatchNormalization* after a given layer? How much dropout should you use? And so on. These
architecture-level parameters are called *hyperparameters* to distinguish them from the parameters of model, which are trained via backpropagation.

In practice, experience machine-learning engineers and researchers build intuition over time as to what works and what doesn't when it comes to these choices - they develop
hyperparameters-tuning skills. You initial decisions are almost always suboptimal, even if you have good intuition. You can refine your choices by tweaking them by hand
and retraining the model repeatedly - that's what machine-learning engineers and researchers spend most of their time doing. But it shouldn't be your job as a human to
fiddle with hyperparameters all day - that is better left to a machine.

You need to search the architecture space and find the best-performing ones empirically. That's what the field of automatic hyperparameter optimization is about: it's an
entire field of research and an important one.

The process of optimizing hyperparameters typically looks like this:
1. Choose a set of hyperparameters automatically.
2. Build the corresponding model.
3. Fit it to your training data, and measure the final performance on the validation data.
4. Choose the next set of hyperparameters to try automatically.
5. Repeat.
6. Eventually, measure performance on your test data.

Training the weights of a model is relatively easy: you compute a loss function on a mini-batch of data and then use the Backpropagation algorithm to move the weights in
the right decision. Updating hyperparameters, on the other hand, is extremely challenging.

Becase these challenges are difficult and the field is still young, we currently only have access to very limited tools to optimize models. Often, it turns out that random
search (choosing hyperparameters to evaluate at random, repeatedly) is the best solution, despite being the most naive one.

Overall, hyperparameter optimization is a powerful technique that is an absolute requirement to get to state-of-the-art models on any task or to win machine-learning
competitions.
#### Model ensembling
Another powerful technique for obtaining the best possible results on a task is *model ensembling.* Ensembling consists of pooling together the predictions of a set of
different models, to produce better options.

Ensembling relies on the assumption that different good models trained independently are likely to be good for *different reasons:* each model looks at slightly different
aspects of the data to make its predictions, getting part of the "truth" but not all of it. The easiest way to pool the predictions of a set of classifiers (to *ensemble
the classifiers*) is to average their predictions at inference time.

This will work only if the classifiers are more or less equally good. If one of them is significantly worse than the others, the final predictions may not be as good as
the best classifier of the group.

A smarter way to ensemble classifiers is to do a weighted average, where the weights are learned on the validation data - typically, the better classifiers are given a
higher weight, and the worse classifiers are given a lower weight. TO search for a good set of ensembling weights, you can use random search or a simple optimization
algorithm such as Nelder-Mead.

The key to making ensembling work is the *diversity* of the set of classifiers. Diversity is strength. Diversity is what makes ensembling work. In machine-learning terms,
if all of your models are biased in the same way, then your ensemble will retain this same bias. If your models are *biased in different ways,* the biases will cancel
each other out, and the ensemble will be more robust and more accurate.

For this reason, you should ensemble models that are *as good as possible* while being *as different as possible.* This typically means using very different architectures
or even different brans of machine-learning approaches.

One thing that is found to work well in practice - but that doesn't generalize to every problem domain - is the use of an ensemble of tree-based methods (such as random
forests or gradient-boosted trees) and deep neural networks.

In recent times, one style of basic ensemble that has been very successful in practice is the *wide and deep* category of models, blending deep learning with shallow
learning. Such models consist of jointly training a deep neural network with a large linear model.
#### Wrapping up
* When building high-performing deep convnets, you'll need to use residual connections, batch normalization, and depthwise separable convolutions.
* Building deep networks requires making many small hyperparameter and architecture choices, which together define how good your model will be. Rather than basing these
choices on intuition or random chance, it's better to systematically search hyperparameter space to find optimal choices.
* Winning machine-learning competitions or otherwise obtaining the best possible results on a task can only be done with large ensembles of models. Ensembling via a
well-optimized weighted average is usually good enough. Remember: diversity is strength.
