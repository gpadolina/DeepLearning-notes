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
tensor.
