## Deep learning for text and sequences
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
