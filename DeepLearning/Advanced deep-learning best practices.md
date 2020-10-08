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
