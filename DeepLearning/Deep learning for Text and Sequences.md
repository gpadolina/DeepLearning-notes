# Deep learning for text and sequences
This chapter explores deep-learning models that can process text (understood as sequences of word or sequences of characters), timeseries, and sequence of data in general.
The two fundamental deep-learning algorithsm for sequence processing are *recurrent neural networks* and *1D convnets*, the one-dimensional version of the 2D convents that
was covered previously.

Applications of these algorithms include:
* Document classification and timeseries classification, such as identifying the topic of an article or the author of a book
* Timeseries comparisons, such as estimating how closely related two documents or two stock tickers are
* Sequence-to-sequence learning, such as decoding an English sentence into French
* Sentiment analysis, such as classifying the sentiment of tweets or movie reviews as positive or negative
* Timeseries forecasting, such as predicting the future weather at a certain location, given recent weather data

Following examples will focus on two narrow tasks: sentiment analysis on the IMDB dataset, a task approached earlier, and temperature forecasting.
## Working with text data
Text is one of the most widespread forms of sequence data. It can be understood as either a sequence of characters or a sequence of words, but it's most common to work at
the level of words.

Deep learning for natural-language processing is pattern recognition applied to words, sentences, and paragraphs, in much the same way that computer vision is pattern
recognition applied to pixels.

Like all other neural networks, deep-learning models don't take as input raw text: they only work with numeric tensors. *Vectorizing* text is the process of transforming
text into numeric tensors. This can be done in multiple ways:
* Segment text into words, and transform each word into a vector.
* Segment text into characters, and transform each character into a vector.
* Extract n-grams of words or characters, and transform each n-gram into a vector. *N-grams* are overlapping groups of multiple consecutive words or characters.

Collectively, the different units into which you can break down text (words, characters, or n-grams) are called *tokens*, and breaking text into such tokens is called
*tokenization.* All text-vectorization processes consist of applying some tokenization scheme and then associating numeric vectors with the generated networks. These vectors,
packed into sequence tensors, are fed into deep neural networks. There are multiple ways to associate a vector with a token. In this section, two will be presented: *one-hot
encoding* of tokens, and *token embedding* (typically used exclusively for words, and called *word embedding*).

#### One-hot encoding of words and characters
One-hot encoding is the most common, most basic way to turn a token into a vector. It consists of associating a unique integer index with every word and then turning this
integer index *i* into a binary vector of size *N* (the size of the vocabulary); the vector is all zeros except for the *i*th entry, which is 1.

Note that Keras has built-in utilities for doing one-hot encoding of text at the word level or character level, starting from raw text data. You should use these utilities,
because they take care of a number of important features such as stripping special characters from strings and only taking into account the *N* most common words in your
dataset.

A variant of one-hot encoding is the so-called *one-hot hashing trick*, which you can use when the number of unique tokens in your vocabulary is too large to handle
explicitly.

The main advantage of this method is that it does away with maintaning an explicit word index, which saves memory and allows online encoding of the data (you can generate
token vectors right away, before you've seen all of the available data). The one drawback of this approach is that it's susceptible to *hash collisions*: two different words
may end up with the same hash, and subsequently any machine-learning model looking at these hashes won't be able to tell the difference between these words.

#### Using word embeddings
Another popular and powerful way to associate a vector with a word is the use of dence *word vectors*, also called *word embeddings*. Whereas the vectors obtained through
one-hot encoding are binary, sparse (mostly made of zeros), and very high-dimensional (same dimensionality as the number of words in the vocabulary), word embeddings are
low-dimensional floating point vectors (that is, dense vectors, as opposed to sparse vectors). Unlike the word vectors obtained via one-hot encoding, word embeddings are
learned from data. It's common to see word embeddings that are 256-dimensional, 512-dimensional, or 1,024-dimensional when dealing with very large vocabularies. Word
embeddings pack more information into far fewer dimensions.

There are two ways to obtain word embeddings:
* Learn word embeddings jointly with the main task you care about (such as document classification or sentiment predictions).
* Load into your model word embeddings that were precomputed using a different machine-learning task than the one you're trying to solve. These are called *pretrained
word embeddings.*

#### Learning word embeddings with the embedding layer
The simplest way to associate a dense vector with a word is to choose the vector at random. The problem with this approach is that the resulting embedding space has no
structure: for instance, the words *accurate* and *exact* may end up with completely different embeddings, even though they're interchangeable in most sentences.

To get a bit more abstract, the geometric relationships between word vectors should reflect the semantic relationships between these words. Word embeddings are meant to map
human language into a geometric space. For instance, in a reasonable embedding space, you would expect synonyms to be embedded into similar word vectors; and in general,
you would expect the geometric distance (such as L2 distance) between any word vectors to relate the semantic distance between the associated words (words meaning different
things are embedded at points far away from each other, whereas realted words are closer).

With the vector representations we chose here, some semantic relationships between these words can be encoded as geometric transformations. For instance, the same vector
allows us to go from *cat* to *tiger* and from *dog* to *wolf*: this vector could be interpreted as the "from pet to wild animal" vector. Similarly, another vector let us
go from *dog* to *cat* and from *wolf* to *tiger*, which could be interpreted as a "from caline to feline" vector.

In real-world word-embedding spaces, common examples of meaningful geometric transformations are "gender" vectors and "plural" vectors. For instance, by adding a "female"
vector to the vector "king," we obtain the vector "queen." By adding a "plural" vector, we obtain "kings."

There is no such a thing as *human language* - there are many different languages, and they aren't isomorphic, because a language is the reflection of a specific culture and
a specific context.
#### Using pretrained word embeddings
Instead of learning word embeddings jointly with the problem you want to solve, you can load embedding vectors from a precomputed embedding space that you know is highly
structured and exhibits useful properties - that captures generic aspects of language structure. The rationale behind using pretrained word embeddings in natural-language
processing is much the same as for using pretrained convnents in image classification: you don't have enough data available to learn truly powerful features on your own,
but you expect the features that you need to be fairly generic - that is, common visual features or semantic features.

There are various precomputed databases of word embeddings that you can download and use in a Keras *Embedding* layer. Word2vec is one of them. This embedding technique is
based on factorizing a matrix of word co-occurence statistics. Its developers have made available precomputed embeddings for millions of English tokens, obtained from
Wikipedia data and Common Crawl data.

#### Putting it all together: from raw text to word embeddings
You'll use a model similar to the one we just went over: embedding sentences in sequences of vectors, flattening them, and training a *Dense* layer on top. But you'll do so
using pretrained word embeddings; and instead of using the pretokenized IMDB data packaged in Keras, you'll start from scratch by downloading the original text data.

#### Preprocessing the embeddings
You'll build an embedding matrix that you can load into an *Embedding* layer. It must be a matrix of shape *(max_words, embedding_dim)*, where each entry *i* contains the
*embedding_dim*-dimensional vector for the word of index *i* in the reference word index built during tokenization.

Additionally, you'll freeze the *Embedding* layer (set its *trainable* attribute to *False*), following the same rationale you're already familiar with in the context of
pretrained convnet features: when parts of a model are pretrained (like your *Embedding* layer) and parts are randomly initialized (like your classifier), the pretrained
parts shouldn't be updated during training, to avoid forgetting what they already know.

You can also train the same model without loading the pretrained word embeddings and without freezing the embedding layer. In that case, you'll learn a task-specific
embedding of the input tokens, which is generally more powerful than pretrained word embeddings when lots of data is available.
## Understanding recurrent neural networks
A major characteristics of all neural networks you've seen so far, such as densely connected networks and convents, is that they have no memory. Such networks are called
*feedforward networks.*

In contrast, as you're readint the present sentence, you're processing it word by word - or rather, eye saccade by eye saccade - while keeping memories of what came before;
this gives you a fluid representation of the meaning conveyed by this sentence. Biological intelligence processes information incrementally while maintaining an internal
model of what it's processing, built from past information and constantly updated as new information comes in.

A *recurrent neural network* (RNN) adopts the same principle, albeit in an extremely simplified version: it processes sequences by iterating through the sequence elements
and maintaning a *state* containing information relative to what it has seen so far. In effect, an RNN is a type of neural network that has an internal loop.

To make these notions of *loop* and *state* clear, let's' implement the forward pass of a toy RNN in Numpy. This RNN takes as input a sequence of vectors, which you'll
encode as a 2D tensor of size *(timesteps, input_features).* It loops over timesteps, and at each timestep, it considers its current state at *t* and the input at *t* (of
shape *(input_features, )*, and combines them to obtain the output at t. For the first timestep, the previous output isn't defined; hence, there is no current state. So,
you'll initialize the state as an all-zero vector called the *initial state* of the network.

In summary, an RNN is a *for* loop that reuses quantities computed during the previous iteration of the loop, nothing more.

#### A recurrent layer in Keras
*SimpleRNN* processes batches of sequences, like all other Keras layers, not a single sequences as in the Numpy example. This means it takes inputs of shape *(batch_size,
timesteps, input_features),* rather than *(timesteps, input_features).*

Like all recurrent layers in Keras, *SimpleRNN* can be run in two different modes: it can return either the full sequences of successive outputs for each timestep (a 3D
tensor of shape *(batch_size, timesteps, output_features)*) or only the last output for each input sequence (a 2D tesnor of shape *(batch_size, output_features)*). These
two modes are controlled by the *return_sequences* constructor argument.

It's sometimes useful to stack several recurrent layers one after the other in order to increase the representational power of a network. In such a setup, you have to get
all of the intermediate layers to return full sequence of outputs.
#### Understanding the LSTM and GRU layers
*SimpleRNN* isn't the only recurrent layer available in Keras. There are two others: *LSTM* and *GRU.* In practice, you'll always use one of these, because *SimpleRNN* is
generally too simplistic to be of real use. *SimpleRNN* has a major issue: although it should theoretically be able to retain at time t information about inputs seen many
timesteps before, in practice, such long-term dependencies are impossible to learn. This is due to the *vanishing gradient problem,* an effect that is similar to what is
observed with non-recurrent networks (feedforward networks) that are many layers deep: as you keep adding layers to a network, the network eventually becomes untrainable.
The *LSTM* and *GRU* layers are designed to solve this problem.

Let's consider  the *LSTM* layer. The underlying Long Short-Term Memory (LSTM) algorithm was developed by Hochreiter and Schmidhuber in 1997; it was the culmination of their
research on the vanishing gradient problem.

Imagine a conveyor belt running parallel to the sequence you're processing. Information from the sequence can jump onto the conveyor belt at any point, be transported to a
later timestep, and jump off, intact, when you need it. This is essentially what LSTM does: it saves information for later, thus preventing older signal from gradually
vanishing during processing.

The specification of an RNN cell determines your hypothesis space - the space in which you'll search for a good model configuration during training - but it doesn't
determine what the cell does; that is up to the cell weights. The same cell with different weights can be doing very different things. So the combination of operations
making up an RNN cell is better interpreted as a set of *constraints* on your search, not as a *design* in an engineering sense.

In summary: you don't need to understand anything about the specific architecture of an *LSTM* cell; as a human, it shouldn't be your job to understand it. Just keep in mind
what the *LSTM* cell is meant to do: allow past information to be reinjected at a later time, thus fighting the vanishing-gradient problem.
#### A concreate LSTM example in Keras
You only specific the output dimensionality of the *LSTM* layer; leave every other argument (there are many) at the Keras defaults. Keras has good defaults, and things
will almost always "just work" without having to spend time tuning parameters by hand.

Analyzing the global, long-term structure of the reviews (what LSTM is good at) isn't helpful for a sentiment-analysis problem. Such a basic problem is well solved by looking
at what words occur in each review, and at what frequency. That's what the first fully connected approach looked at. But there are far more difficult natural
language-processing problems out there, where the strength of LSTM will become apprarent: in particular, question-answering and machine translation.

## Advanced use of recurrent neural networks
We'll review three advanced techniques for improving the performance and generalization power of recurrent neural networks. By the end of the section, you'll know most of
what there is to know about using recurrent networks with Keras. We'll cover the following techniques:
* *Recurrent dropout* - This is a specific, built-in way to use dropout to fight overfitting in recurrent layers.
* *Stacking recurrent layers* - This increases the representational power of the network (at the cost of higher computational loads).
* *Bidirectional recurrent layers* - These present the same information to a recurrent network in different ways, increasing accuracy and mitigating forgetting issues.
#### A basic machine-learning approach
In the same way that it's useful to establish a common-sense baseline before trying machine-learning approaches, it's useful to try simple, cheap machine-learning models
such as small, densely connected networks before looking into complicated and computationally expensive models such as RNNs. 
#### A first recurrent baseline
Let's instead look at the data as what it is: a sequence, where causality and order matter. You'll try a recurrent-sequence processing model - it should be the perfect fit
for such sequence data, precisely because it exploits the temportal ordering of data points.

You'll use the *GRU* layer, developed by Chung et al. Gated recurrent unit (GRU) layers work using the same principle as LSTM, but they're somewhat streamlined and thus
cheaper to run although they may not has as much as representation power as LSTM. This trade-off between computation expensiveness and representational power is seen
everywhere in machine learning.
#### Using recurrent dropout to fight overfitting
You're already familiar with a classic technique for fighting this phenomenon: dropout, which randomly zeros out input units of a layer in order to break happenstance
correlations in the training data that the layer is exposed to. But how to correctly apply dropout in recurrent networks isn't a trivial question.

It has long been known that applying dropout before a recurrent layer hinders learning rather than helping with regularization. The same dropout mask (the same pattern of
dropped units) should be applied at every timestep, instead of a dropout mask that varies randomly from timestep to timestep.
