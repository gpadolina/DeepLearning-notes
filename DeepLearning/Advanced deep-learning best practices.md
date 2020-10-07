## Advanced deep-learning best practices
This explores a number of powerful tools that will bring you closer to being able to develop state-of-the-art models on difficult problems. Using the Keras functional API,
you can build graph-like models, share a layer across different inputs, and use Keras models just like Python functions. Keras callbacks and the TensorBoard browser-based
visualization tool let you monitor models during training.
## Going beyong the Sequential model: the Keras functional API
Until now, all neural networks introduced have been implemented using the *Sequential* model. The *Sequential* model makes the assumption that the network has exactly one
input and exactly one output, and that it consists of a linear stack of layers.
