To use a tool appropriately, you should not only understand what it *can* do but also be aware of what it *can't* do.
## Key concepts
#### Various approaches to AI
First of all, deep learning isn't synonymous with AI or even with machine learning. *Artificial intelligence* is an ancient, broad field that can generally be defined as "all
attempts to automate cognitive processes" - in other words, the automation of thought.

*Machine learning* is a specific subfield of AI that aims at automatically developing programs (called *models*) purely from exposure to training data. This process of turning
data into a program is called *learning.*
#### What makes deep learning special within the field of machine learning
Given training data (in particular, training data appropriately labeled by humans), it's possible to extradct from perceptual data almost anything that a human could extract.
Hence, it's sometimes said that deep learning has *solved perception,* although that's true only for a fairly narrow definition of *perception.*

Due to its unprecedented technical successes, deep learning has singlehandedly brought about the third and by far the largest *AI summer:* a period of intense interested,
investment, and hype in the field of AI.

The hype may recede, but the sustained economic and technological impact of deep learning will remain. In that sense, deep learning could be analogous to the internet: it may
overly hypes up for a few years, but in the longer term it will still be a major revolution that will transform our economy and our lives.
#### How to think about deep learning
The most surprising thing about deep learning is how simple it is. As Feynman once said about the universe, "It's not complicated, it's just a lot of it."

In deep learning, everything is a vector: everything is a *point* in a *geometric space.* Model inputs (text, images, and so on) and targets are first *vectorized:* turned
into an initial input vector space and target vector space. Each layer in a deep-learning model operates one simple geometric transformation on the data that goes through it.
A key characteristics of this geometric transformation is that it must be *differentiable,* which is required in order for us to be able to learn its parameters via gradient
descent.

The full uncrumpling gesture sequence is the complex transformation of the entire model. Deep-learning models are mathematical machines for uncrumpling complicated manifolds
of high-dimensional data.

That's the magic of deep learning: turning meaning into vectors, into geometric spaces, and then incrementally learning complex geometric transformations that map one space
to another.

Neural networks initially emerged from the idea of using graphs as a way to encode meaning, which is why they're named *neural networks;* the surrounding field of research
used to be called *connectionism.* Nowadays the name *neural network* exists purely for historical reasons - it's an extremely misleading name because they're neither neural
nor networks. In particular, neural networks have hardly anything to do with the brain. A more appropriate name would have been *layered representations learning* or
*hiearchical representations learning,* or maybe even *deep differentiable models* or *chained geometric transforms,* to emphasize the fact that continuous geometric space
manipulation is at their core.
#### Key enabling technologies
In the case of deep learning, we can point out the following key factors:
* Incremental algorithmic innovations, first spread over two decades (starting with backpropagation) and then happening increasingly faster as more research effort was poured
into deep learning after 2012.
* The availability of large amounts of perceptual data, which is a requirement in order to realize that sufficiently large models trained on sufficiently large data are all
we need. This is in turn a byproduct of the rise of the consumer internet and Moore's law applied to storage media.
* The availability of fast, highly parallel computation hardware at a low price, especially the GPUs produced by NVIDIA - first gaming GPUs and then chips designed from the
ground up for deep learning. Early on, NVIDIA CEO Jensen Huang took note of the deep-learning boom and decided to bet the company's future on it.
* A complex stack of software layers that makes this computational power available to humans: the CUDA language, frameworks like TensorFlow that do automatic differentiation,
and Keras, which makes deep learning accessible to most people.
#### The universal machine-learning workflow
Having access to an extremely powerful tool for creating models that map any input space to any target space is great, but the difficult part of the machine-learning workflow
is often everything that comes before designing and training such models (and, for production models, what comes after, as well). Understanding the problem domain so as to be
able to determine what to attempt to predict, given what data, and how to measure success, is a prerequisite for any successful application of machine learning, and it isn't
something that advanced tools like Keras and TensorFlow can help you with. As a reminder, here's a quick summary of the typical machine-learning workflow:
1. Define the problem: What data is available, and what are you trying to predict? Will you need to collect more data or hire people to manually label a dataset?
2. Identify a way to reliably measure sucess on your goal. For simple tasks, this may be prediction accuracy, but in many cases it will require sophisticated domain-specific
metrics.
3. Prepare the validation process that you'll use to evaluate your models. In particular, you should define a training set, a validation set, and a test set. The validation-
and test-set labels shouldn't leak into the training data: for instance, with temporal prediction, the validation and test data should be posterior to the training data.
4. Vectorize the data by turning it into vectors and preprocessing it in a way that makes it more easily approachable by a neural network (normalization, and so on).
5. Develop a first model that beats a trivial common-sense baseline, thus demonstrating that machine learning can work on your problem. This may not always be the case!
6. Gradually refine your model architecture by tuning hyperparameters and adding regularization. Make changes based on performance on the validation data only, not the test
data or the training data. Remember that you should get your model to overfit (thus identifying a model capacity level that's greater than you need) and only then begin to
add regularization or downsize your model.
#### Key network architectures
The three families of network architectures that you should be familiar with are *densely connected networks, convolutional networks,* and *recurrent networks.* Each type of
network is meant for a specific input modality: a network architecture (dense, convolutional, recurrent) encodes *assumptions* about the structure of the data: a *hypothesis
space* within which the search for a good model will proceed.

Here's a quick overview of the mapping between input modalities and appropriate network architectures:
* *Vector data* - Densely connected network (*Dense* layers).
* *Image data* - 2D convnets.
* *Sound data (for example, waveform)* - Either 1D convnets (preferred) or RNNs.
* *Text data* - Either 1D convnets (preferred) or RNNs.
* *Timeseries data* - Either RNNs (preferred) or 1D convnets.
* *Other types of sequence data* - Either RNNs or 1D convnets. Prefer RNNs if data ordering is strongly meaningful (for example, for timeseries, but not for text).
* *Video data* - Either 3D convnets (if you need to capture motion effects) or a combination of a frame-level 2D convnet for feature extraction followed by either an RNN or
a 1D convnet to process the resulting sequences.
* *Volumetric data* - 3D convnets.
#### Densely connected networks
A densely connected network is a stack of *Dense* layers, meant to process vector data (batches of vectors). Such networks assume no specific structure in the input features:
they're called *densely connected* because the units of a *Dense* layer are connected to every other unit.

Remember: to perform *binary classification,* end your stack of layers with a *Dense* layer with a single unit and a *sigmoid* activation, and use *binary_crossentropy* as
the loss. Your targets should be either 0 or 1.
