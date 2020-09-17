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