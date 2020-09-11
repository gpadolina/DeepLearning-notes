# Mathematical building blocks of neural networks

The cord building block of neural networks is the *layer*, a data-preprocessing module that you can think of as a filter for data. Some data goes in and it comes out
in a more useful form. Specifically, layers extract *representations* out of the data fed into them - hopefully, representation that are more meaningful for the problem
at hand.

To make a network ready for training, we need to pick three more things, as part of the *compilation* step:
