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
