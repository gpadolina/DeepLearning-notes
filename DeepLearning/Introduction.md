# Deep Learning

## Artificial intelligence, machine learning, and deep learning

#### Artificial intelligence
The effort to automate intellectual tasks normally performed by humans. As such, AI is a general field that encompasses machine learning and deep learning, but that
also includes many more approaches that don't involve any learning.
#### Machine learning
Machine learning arises from this question: could a computer go beyong "what we know how to order it to perform" and learn on its own how to perform specified task?
Could a computer surprise us? Rather than programmers crafting data-processing rules by hand, could a computer automatically learn these rules by looking at data?

With machine learning, humans input data as well as the answers expected from the data, and out come the rules. These rules can then be applied to new data to produce
original answers. A machine-learning system is trained rather than explicitly programmed.

Machine learning is tightly related to mathematical statistics, but it differs from statistics in several important ways. Unlike statistics, machine learning tends
to deal with large, complex datasets for which classical statistical analysis such as Bayesian would be impractical. As a result, machine learning, and especially
deep learning, exhibits comparatively little mathematical theory and is engineering oriented.

The central problem in machine learning and deep learning is to meaningfully transform data. In other words, to learn useful representations of the input data at
hand - representations that get us closer to the expected output. Representation is a different way to look at data - to represent or encode data. Learning in the
context of machine learning, describes an automatic search process for better representations.

Technically, machine learning is searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback
signal.
#### Deep learning
Deep learning is a specific subfield of machine learning: a new tkae on learning representations from data that puts on emphasis on learning successive layers of
increasingly meaningful representations. The *deep* in *deep learning* isn't a reference to any kind of deeper undestanding achieved by the approach; rather, it stands
for this idea of successive layers of representations. How many layers contribute to a model of the data is called the *depth* of the model.

Other approaches of machine learning tend to focus on learning only one or two layers of representations of the data; hence, they're sometimes called *shallow learning*.

In deep learning, these layered representations are learned via models called *neural networks*, structured in literal layers stacked on top of each other. The term
*neural network* is a reference to neurobiology, but although some of the central concepts in deep learning were developed in part by drawing inspiration from our
understanding of the brain, deep-learning models are not models of the brain.

You can think of a deep network as a multistage information-distillation , where information goes through successive filters and comes out increasingly *purified*.

Technically, deep learning is a multistage way to learn data representations. It's a simple idea - but as it turns out, very simple mechanisms, sufficiently scaled,
can end up looking like magic.
#### Understanding how deep learning works
The specification of what a layer does to its input data is stored in the layer's *weights*, which in essence are a bunch of numbers. In technical terms, the transformation
implemented by a layer is *parameterized* by its weights. In this context, *learning* means finding a set of values for the weights of all layers in a network, such
that the network will correctly map example inputs to their associated targets. A deep neural network can contain tens of millions of parameters.

To control the output of a neural network, you need to be able to measure how far this output is from what you expected. This is the job of the *loss function* of the
network, also called the *objective function*. The loss function takes the predictions of the network and the true target and computes a distance score, capturing how
well the network has done on this example.

The fundamental trick in deep learning is to use this score as a feedback signal to adjust the value of the weights a little, in a direction that will lower the loss
score for the current example. This adjustment is the job of the *optimizer*, which implements what's called the *Backpropagation* algorithm: the central algorithm in
deep learning.
## Before machine learning

#### Probabilistic modeling
*Probabilistic modeling* is the application of the principles of statistics to data analysis. It was one of the earliest forms of machine learning and it's still
widely used to this day. One of the best-known algorithms is the Naive Bayes algorithm. Naive Bayes is a type of machine learning classifier based on applying Bayes'
theorem while assuming that the features in the input data are all independent.

A closely related model is the *logistic regression*. Don't be misled by its name - logreg is a classification algorithm rather than a regression algorithm. It's often
the first thing a data scientist will try on a dataset to get a feel for the classification task at hand.
#### Kernel methods
*Kernel methods* are a group of classification algorithms, the best known of which is the *support vector machine* (SVM). SVMs aim at solving classification problems
by finding good *decision boundaries* between two points of belonging to two different categories.

To find good decision hyperplanes in the new representation space, you don't have to explicitly compute the coordinates of your points in the new space; you just need
to compute the distance between pairs of points in that space, which can be done efficiently using a *kernel function*. A kernel function is a computationally
tractable operation that maps any two points in your inital space to the distance between these points in your target representation space, completely bypassing the
explicit computation of the new representation.
#### Decision trees, random forests, and gradient boosting machines
*Decision trees* are flowchart-like structures that let you classify input data points or predict output values given inputs. They're easy to visualize and interpret.

The *random forest* algorith introduced a robust, practical take on decision-tree learning that involves building a large number of specialized decision trees and
then ensembling their outputs. Random forests are applicable to a wide range of problems - you could say that they're almost always the second-best algorithm for any
shallow machine-learning task.

A *gradient boosting machine*, much like a random forest, is a machine-learning technique based on ensembling weak prediction models, generally decision trees. It
uses *gradient boosting*, a way to improve any machine-learning model by iteratively training new models that specialize in addressing the weak points of the previous
models. Applied to decision trees, the use of the gradient boosting technique results in models that strictly outperform random forests most of the time, while having
similar properties. It may be one of the best algorithm for dealing with nonperceptual data.
