# Mathematical building blocks of neural networks

The cord building block of neural networks is the *layer*, a data-preprocessing module that you can think of as a filter for data. Some data goes in and it comes out
in a more useful form. Specifically, layers extract *representations* out of the data fed into them - hopefully, representation that are more meaningful for the problem
at hand.

To make a network ready for training, we need to pick three more things, as part of the *compilation* step:
* *Loss function* - How the network will be able to measure its performance on the training data, and thus how it will be able to steer itself in the right direction.
* *Optimizer* - The mechanism through which the network will update itself based on the data it sees and its loss function.
* *Metrics to monitor during training and testing*
