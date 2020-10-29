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

To perform *single-label classification* (where each sample has exactly one class, no more), end your stack of layers with a *Dense* layer with a number of units equal to the
number of classes, and a *softmax* activation. If your targets are one-hot encoded, use *categorical_crossentropy* as the loss; if they're integers, use
*sparse_categorical_crossentropy.*

To perform *multilabel categorical classification* (where each sample can have several classes), end your stack of layers with a *Dense* layer with a number of units equal
to the number of classes and a *sigmoid* activation, and use *binary_crossentropy* as the loss. Your targets should be k-hot encoded.

To perform *regression* toward a vector of continuous values, end your stack of layers with a *Dense* layer with a number of units equal to the number of values you're trying
to predict (often a single one, such as the price of a house), and no activation. Several losses can used for regression, most commonly *mean_squared_error* (MSE) and
*mean_absolute_error* (MAE).
#### Convnets
Convolution layers look at spatially local patterns by applying the same geometric transformation to different spatial locations *(patches)* in an input tensor. You can use
the *Conv1D* layer to process sequences (especially text - it doesn't work as well on timeseries, which often don't follow the translation-invariance assumption), the
*Conv2D* layer to process images, and the *Conv3D* layers to process volumes.

Convnets are often ended with either a *Flatten* operation or a global pooling layer, turning spatial feature maps into vectors, followed by *Dense* layers to achieve
classification or regression.

Note that it's highly likely that regular convolutions will soon be mostly (or completely) replaced by an equivalent but faster and representationally efficient alternative:
the *depthwise separable convolution* (*SeparableConv2D* layer). This is true for 3D, 2D, and 1D inputs. When you're building a new network from scratch, using depthwise
separable convolutions is definitely the way to go.
#### RNNs
*Recurrent neural networks* (RNNs) work by processing sequences of inputs one timestep at a time and maintaining a *state* throughout (a state is typically a vector or set
of vectors: a point in a geometric space of states).

Three RNN layers are available in Keras: *SimpleRNN, GRU,* and *LSTM.* For most practical purposes, you should use either *GRU* or *LSTM. LSTM* is the more powerful of the
two but is also more expensive; you can think of *GRU* as a simpler, cheaper alternative to it.

If you aren't stacking any further RNN layers, then it's common to return only the last output, which contains information about the entire sequence.
#### The space of possibilities
Mapping vector data to vector data
* *Predictive healthcare* - Mapping patient medical records to predictions of patient outcomes
* *Behavioral targeting* - Mapping a set of website attributes with data on how long a user will spend on the website
* *Product quality control* - Mapping a set of attributes relative to an instance of a manufactured product with the probability that the product will fail by next year
Mapping image data to vector data
* *Doctor assistant* - Mapping slides of medical images with a prediction about the presence of a tumor
* *Self-driving vehicle* - Mapping car dash-cam video frames to steering wheel angle commands
* *Board game AI* - Mapping GO and chess boards to the next player move
* *Diet helper* - Mapping pictures of a dish to its calorie count
* *Age prediction* - Mapping selfies to the age of the person
Mapping timeseries data to vector data
* *Weather prediction* - Mapping timeseries of weather data in a grid of locations of weather data the following week at a specific location
* *Brain-computer interfaces* - Mapping timeseries of magnetoencephalogram (MEG) data to computer commands
* *Behavioral targeting* - Mapping timeseries of user interactions on a website to the probability that a user will buy something
Mapping text to text
* *Smart reply* - Mapping emails to possible one-line replies
* *Answering questions* - Mapping general-knowledge questions to answers
* *Summarization* - Mapping a long article to a short summary of the article
Mapping images to text
* *Captioning* - Mapping images to short captions describing the contents of the images
Mapping text to images
* *Conditioned image generation* - Mapping a short text description to images matching the description
* *Logo generation/selection* - Mapping the name and description of a company to the company's logo
Mapping images to images
* *Super-resolution* - Mapping downsized images to higher-resolution versions of the same images
* *Visual depth sensing* - Mapping images of indoor environments to maps of depth predictions
Mapping images and text to text
* *Visual QA* - Mapping images and natural-language questions about the contents of images to natural-languages answers
Mapping video and text to text
* *Video QA* - Mapping short videos and natural-language questions about the contents of videos to natural-language answers

## The limitations of deep learning
The space of applications that can be implemented with deep learning is nearly infinite. And yet, many applications are completely out of reach for current deep-learning
techniques - even given vast amounts of human-annotated data. In general, anything that requires reasoning - like programming or applying the scientific method - long-term
planning, and algorithmic data manipulation is out of reach for deep-learning models, no matter how much data you throw at them. Even learning a sorting algorithm with a
deep neural network is tremendously difficult.

This is because a deep-learning model is just a *chain of simple, continuous geometric transformations* mapping one vector space into another. All it can do is map one data
manifold X into another manifold Y, assuming the existence of a learnable continuous transform from X to Y. A deep-learning model can be interpreted as a kind of program;
but, inversely, *most programs can't be expressed as deep-learning models.*

Scaling up current deep-learning techniques by stacking more layers and using more training data can only superficially palliate some of these issues. It won't solve the more
fundamental problems that deep-learning models are limited in what they can represent and that most of the programs you may wish to learn can't be expressed as a continuous
geometric morphing of a data manifold.
#### The risk of anthropomorphizing machine-learning models
One real risk with contemporary AI is misinterpreting what deep-learning models do and overestimating their abilities. A fundamental feature of humans is our *theory of mind:*
our tendency to project intentions, beliefs, and knowledge on the things around us.

In short, deep-learning models don't have any understanding of their input - at least, not in a human sense. Our own understanding of images, sounds, and language is grounded
in our sensorimotor experience as humans. Machine-learning models have no access to such experiences and thus can't understand their inputs in a human relatable way.

As a machine-learning practitioner, always be mindful of this, and never fall into the trap of believing that neural networks understand the task they perform - they don't,
at least not in a way that would make sense to us.
#### Local generalization vs. extreme generalization
Humans are capable of far more than mapping immediate stimuli to immediate responses, as a deep network, or maybe an insect, would. We can merge together known concepts to
represent something we've never experienced before. This ability to handle hypotheticals, to expand our mental model space far beyond what we can experience directly - to
perform *abstraction* and *reasoning* - is arguably the defining chracteristic of human recognition. *Extreme generalization:* an ability to adapt to novel,
never-before-experienced situations using lilttle data or even no new data at all.

This stands in sharp contrast with what deep nets do, which is *local generalization.* On the other hand, humans are able to learn safe behaviors without having to die even
once - again, thanks to our power of abstract modeling of hypothetical situations.

In short, despite our progress on machine perception, we're still far from human-level AI. Our models can only perform local generalization, adapting to new situations that
must be similar to past data, whereas human cognition is capable of extreme generalization, quickly adapting to radically novel situations and planning for long-term future
situations.
#### Wrapping up
Here's what you should remember: the only real success of deep learning so far has been the ability to map space X to space Y using a continuous geometric transform, given
large amounts of human-annotated data. Doing this well is a game-changer for essentially every industry, but it's still a long way from human-level AI.

## The future of deep learning
At a high level, these are the main directions where deep learning is heading:
* *Models closer to general-purpose computer programs,* built on top of far richer primitives than the current differentiable layers. This is how we'll get to *reasoning*
and *abstraction,* the lack of which is the fundamental weakness of current models.
* *New forms of learning that make the previous point possible,* allowing models to move away from differentiable transforms.
* *Models that require less involvement from human engineers.* It shouldn't be your jobs to tune knobs endlessly.
* *Greater, systematics reuse of previously learned features and architectures,* such as meta-learning systems using reusable and modular program subroutines.
#### Models are programs
As noted in the previous section, a necessary transformational development that we can expect in the field of machine learning is a move away from models that perform purely
*pattern recognition* and can only achieve *local generalization,* toward models capable of *abstraction* and *reasoning* that can achieve *extreme generalization.*

It's important to note that RNNs have slightly fewer limitations than feedforward networks. That's because RNNs are a bit more than mere geometric transformations: they're
geometric transformations *repeatedly applied inside a for loop.* The temporal for loop is itself hardcoded by human developers: it's a built-in assumption of the network.

A realted subfield of AI that I think may be about to take off in a big way is *program synthesis,* in particular neural program systhesis. Program synthesis consists of
automatically generating simple programs by using a search algorithm (possibly genetic search, as in genetic programming) to explore a large space of possible programs. The
search stops when a program is found that matches the required specifications, often provided as a set of input-output pairs.
#### Automated machine learning
Currently, most of the job of a deep-learning engineer consists of munging data with Python scripts and then tuning the architecture and hyperparameters of a deep network
at length to get a working model - or even to get a state-of-the-art model, if the engineer is that ambitious. Needless to say, that isn't an optimal setup.
#### Lifelong learning and modular subroutine reuse
A remarkable observation has been made repeatedly in recent years: training the *same* model to do several loosely connected tasks at the same time results in a model that's
*better at each task.*

This is fairly intuitive: there's always *some* information overlap between seemingly disconnected tasks, and a joint model has access to a greater amount of information
about each individual task than a model on that specific task only.

When the system finds itself developing similar program subroutines for several different tasks, it can come up with an abstract, reusable version of the subroutine and store
it in the global library. Such a process will implement *abstraction:* a necessary component for achieving extreme generalization.
