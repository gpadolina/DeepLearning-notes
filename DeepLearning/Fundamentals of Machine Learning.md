# Fundamentals of machine learning

## Four branches of machine learning
The three specific types of machine-learning problems: binary classification, multiclass classification, and scalar regression are all instances of *supervised learning*,
where the goal is to learn the relationship between training inputs and training targets.

#### Supervised learning
This is by far the most common case. Although supervised learning mostly consists of classfication and regression, there are more exotic variants as well.
* *Sequence generation* - Given a picture, predict a caption describing it.
* *Syntax tree prediction* Given a sentence, predict its decomposition into a syntax tree.
* *Object detection* - Given a picture, draw a bounding box around certain objects inside the picture.
* *Image segmentation* - Given a picture, draw a pixel-lvel mask on a specific object.
#### Unsupervised learning
Consists of finding interesting transformation of the input data without the help of any targets. Unsupervised learning is the bread and butter of data analytics and it's
often a necessary step in better understanding a dataset before attempting to solve a supervised-learning problem. *Dimensionality reduction* and *clustering* are
well-known categories of unsupervised learning.
#### Self-supervised learning
This is a specific instance of supervised learning, but it's different enough that it deserves its own category. Self-supervised learning is supervised learning without
human-annotated labels. There are still labels involved, but they're generated form the input data, typically using a heuristic algorithm.
#### Reinforcement learning
In reinforcement learning, an *agent* receives information about its environment and learns to choose actions that will maximize some reward.

## Evaluating machine-learning models
The reason not to evaluate the models on the same data they were trained on quickly became evident: after just a few epochs, all three models began to *overfit*. That is,
their performance on never-before-seen data started stalling or worsening compared to their performance on the training data - which always improves as training
progresses.
In machine learning, the goal is to achieve models that *generalize* - that perform on never-before-seen data and overfitting is the central obstacle. You can only control
that which you can observe, so it's crucial to be able to reliably measure the generalization power of your model.
#### Traning, validation, and test sets
Evaluating a model always boils down to splitting the available data into three sets: training, validation, and test. You train on the training data and evaluate your model
on the validation data. Once your model is ready for prime time, you test it one final time on the test data.

Why not have two sets? The reason is that developing a model always involves tuning its configuration: for example, choosing the number of layers or the size of the layers
(called the *hyperparameters* of the model, to distinguish them from the *parameters*, which are the network's weights). You do this tuning by using as a feedback signal
the performance of the model on the validation data. In essence, this tuning is a form of *learning*: a search for a good configuration in some parameter space. Tuning the
configuration of the model based on its performance on the validation set can quickly result in *overfitting to the validation set*, even though your model is never directly
trained on it.

Central to this phenomenon is the notion of *information leaks*. Every time you tune a hyperparameter of your model based on the model's performance on the validation set,
some information about the validation data leaks into the model.

At the end of the day, you'll end up with a model that performs artifically well on the validation data, because that's why you optimized it for. You care about performance
on completely new data, not the validation data, so you need to use a completely different, never-before-seen dataset to evaluate the model: the test dataset.

Let's review three classic  evaluation recipes: simple hold-out validation, K-fold avalidation, and iterated K-fold validation with shuffling.
* *Simple hold-out validation*: Set apart some fraction of your data as your test set. Train on the remaining data, and evaluate on the test set. This is the simplest
evaluation protocol and it suffers from one flaw: if little data is available, then your validation and test sets may contain too few samples to be statistically
representative of the data at hand.
* *K-fold validation*: With this approach, you split your data into K partitions of equal size. For each partition i, train a model on the remaining K-1 partitions, and
evaluate it on partition i. Your final score is then the averages of the K scores obtained. This method is helpful when the performance of your model shows significant
variance based on your train-test split. Like hold-out validation, this method doesn't exempt you from using a distinct validation set for model calibration.
* *Iterated K-fold validation with shuffling*: This one is for situations in which you have relatively little data available and you need to evaluate your model as
precisely as possible. It consists of applying K-fold validation multiple times, shuffling the data every time before splitting it K ways.
## Data preprocessing, feature engineering, and feature learning
How do you prepare the input data and targets before fedding them into a neural network? Many data-preprocessing and feature-engineering techniques are domain specific.
#### Data preproceesing for neural networks
Data preprocessing aims at making the raw data at hand more amenable to neura networks. This includes vectoriation, normalization, handling missing values, and feature
extraction.
* *Vectorization* - All inputs and targets in a neural network must be tensors of floating-point data. Whatever data you need to process - sound, images, text - you must
first turn into tensors, a step called *data normalization*.
* *Value normalization* - Before you feed data into a network, you have to normalize each feature independently so that it has a standard deviation of 1 and a mean of 0.
In general, it isn't safe to feed into a neural network data that takes relatively large values or data that is heterogeneous. Doing so can trigger large gradient updates
that will prevent the network from converging. To make learning easier for your networks, your data should *take small values* - typically in the 0-1 range, and *be
homogeneous* - all features should take values in roughly the same range.
* *Handling missing values* - In general, with neural networks, it's safe to input missing values as 0, with the condition that 0 isn't already a meaningful value. The
network will learn from exposure to the data that the value 0 means *missing data* and will start ignoring the value. Note that if you're expecting missing values in the
test data, but the network was trained on data without any missing values, the network won't have learned to ignore missing values.
#### Feature engineering
*Feature engineering* is the process of using your own knowledge about the data and about the machine-learning algorithm at hand to make the algorithm work better by
applying hardcoded transformations to the data before it goes into the model.

The essence of feature engineering: making a problem easier by expressing it in a simpler way. It usually requires understanding the problem in depth. Fortunately, modern
deep learning removes the need for most feature engineering because neural networks are capable of automatically extracting useful features from raw data.
## Overfitting and underfitting
Overfitting happens in every machine-learning problem. The fundamental issue in machine learning is the tension between optimization and generalization. *Optimization*
refers to the process of adjusting a model to get the best performance possible on the training data (the *learning* in *machine learning*), whereas *generalization*
refers to how well the trained model performs on data it has never seen before. The goal of the game is to get good generalization, but you don't control generalization;
you can only adjust the model based on its training data.

At the beginning of training, optimization and generalization are correlated: the lower the loss on training data, the lower the loss on test data. While this is happening,
your model is said to be *underfit.* But after a certain number of iterations on the training data, generalization stops improving, and validation metrics stall and then
begin to degrade: the model is starting to overfit.

To prevent a model from learning misleading or irrelevant patterns found in the training data, *the best solution is to get more training data.* All model trained on more
data will naturally generalize better. The processing of fighting overfitting this way is called *regularization.*
#### Reducing the network's size
The simplest way to prevent overfitting is to reduce the size of the model: the number of learnable parameters in the model, which is determined by the number of layers and
the number of units per layer. In deep learning, the number of learnable parameters in a model is often referred to as the model's *capacity.* Keep in mind that deep learning
models tend to be good at fitting to the training data, but the real challenge is generalization, not fitting.

At the same time, use models that have enough parameters that they don't underfit: your model shouldn't be starved for memorization resources. There is a compromise to be
found between *too much capacity and not enough capacity.*

The general workflow to find an appropriate model size is to start with relatively few layers and parameters, and increase the size of the layers or add new layers until you
see diminishing returns with regard to validation loss.
#### Adding weight regularization
You may be familiar with the principle of *Occam's razor:* given two explanations for something, the explanation most likely to be correct is the simplest one - the one
that makes fewer assumptions. This idea also applies to the models learned by neural networks: given some training data and a network architecture, multiple sets of weight
values (multiple models) could explain the data. Simpler models are less likley to overfit than complex ones.

A *simple model* in this context is a model where the distribution of parameter values has less entropy (or a model with fewer paramters). Thus a common way to mitigate
overfitting is to put constaints on the complexity of a network by forcing its weights to take only small values, which makes the distribution of weight values more
*regular.* This is called *weight regularization,* and it's done by adding to the loss function of the network a *cost* associated with having large weights.
* *L1 regularization* - The cost added is proportional to the *absolute value of the weights coefficients* (the *L1 norm* of the weights).
* *L2 regularization* - The cost added is proportional to the *square of the value of the weight coefficients* (the *L2 norm* of the weights). L2 regularization is also
called *weight decay* in the context of neural networks. 
In Keras, weight regularization is added by passing *weight regularizer instances* to layers as keyword arguments.
#### Adding dropout
*Dropout* is one of the most effective and most commonly used regularization techniques for neural networks. Dropout, applied to a layer, consists of randomly *dropping out*
(setting to zero) a number of output features of the layer during training. Let's say a given layer would normally return a vector [0.2, 0.5, 1.3, 0.8, 1.1] for a given
input sample during training. After applying dropout, this vector will have a few zero entries distributed at random: for example, [0, 0.5, 1.3, 0, 1.1]. The *dropout
rate* is the fraction of the features that are zeroed out; it's usually set between 0.2 and 0.5. At test time, no units are dropped out; instead, the layer's output values
are scaled down by a factor equal to the dropout rate, to balance for the fact that more units are active than at training time.

This technique may seem strange and arbitrary. Why would this help reduce overfitting? Hinton says he was inspired by a fraud-prevention mechanism used by banks. In his own
words, "I went to my bank. The tellers kept changing and I asked one of them why. He said he didn't know but they got moved around a lot. I figured it must be because it
would require cooperation between employees to successfully defraud the bank. This made me realize that randomly removing a different subset of neurons on each example would
prevent conspiracies and thus reduce overfitting." The core idea is that introducing noise in the output values of layer can break up happenstance that aren't significant,
which the network will start memorizing if no noise is present.

In Keras, you can introduce dropout in a network via the *Dropout* layer, which is applied to the output of the layer right before it.

To recap, these are the most common ways to prevent overfitting in neural networks:
* Get more training data.
* Reduce the capacity of the network.
* Add weight regularization.
* Add dropout.
## The universal workflow of machine learning
The blueprint ties together the concepts in this chapter: problem definition, evaluation, feature engineering, and fighting overfitting.
#### Defining the problem and assembling a dataset
First, you must define the problem at hand:
* What will your input be> What you trying to predict?
* What type of problem are you facing? Is it binary classifications? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification?
Somehing else, like clustering, generation, or reinforcement learning?
You can't move to the next stage until you know what your inputs and outputs are, and what data you'll use. Be aware of the hypotheses you make at this stage.

Not all problems can be solved; just because you've assembled examples of inputs X and targets Y doesn't mean X contains enough information to predict Y. One class of
unsolvable problems you should be aware of is *nonstationary problems.*

What you're trying to model changes over time. In this case, the right move is to constantly retrain your model on data from the recent past, or gather data at a timescale
where the problem is stationary. For a cyclical problem like clothes buying, a few years' worth of data will suffice to capture seasonal variation - but remember to make
the time of the year an input of your model.

Using machine learning trained on past data to predict the future is making the assumption that the future will behave like the past. That often isn't the case.
#### Choosing a measure of success
To control something, you need to be able to observe it. To achieve success, you must define what you mean by success - accuracy? Precion and recall? Customer-retention rate?
Your metric for success will guide the choice of a loss function: what your model will optimize.

For balanced-classifcation problems, where every class is equally likely, accuracy and *area under the receiver operating characteristics curve* (ROC AUC) are common metrics.
For class-imblanced problems, you can use precision and recall. For ranking problems or multilable classfication, you can use mean average precision.
#### Deciding on an evaluation protocol
* *Maintaining a hold-out validation set* - The way to go when you have plenty of ata
* *Doing K-fold cross-validation* - The right choice when you have too few samples for hold-out validation to be reliable
* *Doing iterated K-fold validation* - For performing highly accurate model evaluation when litte data is available
#### Preparing your data
Once you know what you're training on, what you're optimizing for, and how to evaluate your approach, you're almost ready to begin training models. But first, you should
format your data in a way that can be fed into a machine-learning model - here, we'll assume a deep neural network.
#### Developing a model that does better than a baseline
Your goal at this stage is to achieve *statistical power:* that is, to develop a small model that is capable of beating a dumb baseline. Remember that you make two
hypotheses:
* You hypothesize that your outputs can be predicted given your inputs.
* You hypothesize that the available data is sufficiently informative to learn the relationship between inputs and outputs.
You need to make three key choices to build your first working model:
* *Last-layer activation* - This establishes useful constraints on the network's output.
* *Loss function* - This should match the type of problem you're trying to solve.
* *Optimization configuration* - What optimizer will you use? What will its learning rate be? In most cases, it's safe to go with *rmsprop* and its default learning rate.

Regarding the choice of a loss function, note that it isn't always possible to directly optimize for the metric that measures success on problem. Sometimes there is no
easy way to turn a metric into a loss function; loss functions, after all, need to be computable given only a mini-match of data and must be differentiable. For instance,
the widely used classification metric ROC AUC can't be directly optimized. Hence, in classification tasks, it's common to optimize for a proxy metric of ROC AUC, such as
crossentropy. In general, you can hppe that the lower the crossentropy gets, the higher the ROC AUC will be.

| **Problem type** | **Last-layer activation** | **Loss function** |
| --- | --- | --- |
| Binary classfication | sigmoid | binary_crossentropy |
| Multiclass, single-label classfication | softmax | categorical_crossentropy |
| Multiclass, multilabel classification | sigmoid | binary_crossentropy |
| Regression to arbitrary values | None | mse |
| Regression to values between 0 and 1 | sigmoid | mse or binary_crossentropy |

#### Scaling up: developing a model that overfits
Once you've obtained a model that has statistical power, the question becomes, is your model sufficiently powerful? Does it have enough layers and parameters to properly
model that problem at hand? Remember that the universal tension in machine learning is between optimization and generalization; the ideal model is one that stands right at
the border between underfitting and overfitting; between uncercapacity and overcapacity.

To figure out how big a model you'll need, you must develop a model that overfits.
* Add layers.
* Make the layers bigger.
* Train for more epochs.

Always monitor the training loss and validation loss, as well as the training and validation values for any metrics you care about. When you see that the model's performance
on the validation data begins to degrade, you've achieved overfitting. The next stage is to start regularizing and turning the model, to get as close as possible to the ideal
model that neighter underfits nor overfits.
#### Regularizing your model and tuning your hyperparameters
This step will take the most time: you'll repeatedly modify your model, train it, evaluate on your validation data, modify it again, and repeat, until the model is as good as
it can get. These are some things you should try:
* Add dropout.
* Try different architectures: add or remove layers.
* Add L1 and/or L2 regularization.
* Try different hyperparameters to find the optimal configuration.
* Optionally, iterate on feature engineering: add new features or remove features that don't seem to be informative.

Be mindful of the following: every time you use feedback from your validation process to tune your model, you leak information about the validation process into the model.

If it turns out that the performance on the test set is significantly worse than the performance measured on the validation data, this may mean either that your validation
procedure wasn't reliable after all, or that you began overfitting to the validation data while tuning the parameters of the model. In this case, you may want to switch to
a more reliable evaluation protocol such as iterated K-fold validation.
