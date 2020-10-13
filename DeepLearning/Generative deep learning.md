 ## Generative deep learning
 
 The artistic productions we've seen from AI so far have been fairly low quality. AI isn't anywhere close to rivaling human screenwriters, painters, and composers. But
 replacing humans was always beside the point: artificial intelligence isn't about replacing our own intelligence with something else, it's about bringing into our lives
 and work *more* intelligence - intelligence of a different kind. In many field, but especially in creative ones, AI will be used by humans as a tool to augment their own
 capabilities: more *augmented* intelligence than *artificial* intelligence.

Our perceptual modalities, our language, and our artwork all have statistical structure. Learning this structure is what deep-learning algorithms excel at. Machine-learning 
models can learn the statistical *latent space* of images, music, and stories, and they can then *sample* from this space, creating new art- works with characteristics 
similar to those the model has seen in its training data.

## Text generation with LSTM
#### How do you generate sequence data?
The universal way to generate sequence data in deep learning is to train a network (usu- ally an RNN or a convnet) to predict the next token or next few tokens in a sequence,
using the previous tokens as input. For instance, given the input “the cat is on the ma,” the network is trained to predict the target *t*, the next character. 

As usual when working with text data, *tokens* are typically words or characters, and any network that can model the probability of the next token given the previous ones is
called a *language model.* A language model captures the *latent space* of language: its statistical structure.

Once you have such a trained language model, you can *sample* from it (generate new sequences): you feed it an initial string of text (called *conditioning data*), ask it to
generate the next character or the next word (you can even generate several tokens at once), add the generated output back to the input data, and repeat the process many times.

In the example we present in this section, you’ll take a LSTM layer, feed it strings of N characters extracted from a text corpus, and train it to predict character N + 1. 
The output of the model will be a softmax over all possible characters: a probability distribution for the next character. This LSTM is called a *character-level neural 
language model.*
