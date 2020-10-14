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
#### The importance of the sampling strategy
When generating text, the way you choose the next character is crucially important. A naive approach is *greedy sampling,* consisting of always choosing the most likely next
character. But such an approach results in repetitive, predictable strings that don't look like coherent language. A more interesting approach makes slightly more suprising
choices: it introduces randomness in the sampling process, by sampling from the probability distribution for the next character. This is called *stochastic sampling* (recall
that *stochasticity* is what we call *randomness* in this field).

Sampling probabilistically from the softmax output of the model is neat:  it allows even unlikely characters to be sampled some of the time, generating more interesting
looking sentences and sometimes showing creativity by coming up with new, realistic sounding words that didn't occur in the training data. But there's one issue with this
strategy: it doesn't offer a way to *control the amount of randomness* in the sampling process.

Why would you want more or less randomness? Consider an extreme case: pure random sampling, where you draw the next character from a uniform probability distribution, and
every character is equally likely. This scheme has maximum randomness; in other words, this probability distribution has maximum entropy.

Less entropy will give the generated sequences a more predictable structure (and thus they will potentially be more realistic looking), whereas more entropy will result
in more suprising and creative sequences.

In order to control the amount of stochasticity in the sampling process, we'll introduce a parameter called the *softmax temperature* that characterizes the entropy of the
probability distribution used for sampling: it characterizes how suprising or predictable the choice of the next character will be.
#### Implementing character-level LSTM text generation
Note that recurrent neural networks aren't the only way to do sequence data generation; 1D convnets also have proven extremely successful at this task in recent times.

Because your targets are one-hot encoded, you'll use *categorical_crossentropy* as the loss to train the model.
#### Training the language model and sampling from it
Given a trained model and a seed text snippet, you can generate new text by doing the following repeatedly:
1. Draw from the model a probability distribution for the next character, given the generated text available so far.
2. Reweight the distribution to a certain temperature.
3. Sample the next character at random according to the reweighted distribution.
4. Add the next character at the end of the available text.
#### Wrapping up
* You can generate discrete sequence data by training a model to predict the next token(s), given previous tokens.
* In the case of text, such a model is called a *language model.* It can be based on either words or characters.
* Sampling the next token requires balance between adhering to what the model judges likely, and introducing randomness.
* One way to handle this is the notion of softmax temperature. Always experiment with different temperatures to find the right one.
## DeepDream
*DeepDream* is an artistic image-modification technique that uses the representations learned by convolutional neural networks. It was first released by Google in the
summer of 2015, as an implementation written using the Caffe deep-learning library (this was several months before the first public release of TensorFlow).

The DeepDream algorithm is almost identical to the convnet filter-visualization technique introduced in Deep Learning for Computer Vision, consisting of running a convnet
in reverse: doing gradient ascent on the input to the convnet in order to maximize the activation of a specific filter in an upper layer of the convnet. DeepDream uses this
same idea, with a few simple differences:
* With DeepDream, you try to maximize the activation of entire layers rather than that of a specific filter, thus mixing together visualization of large numbers of features
at once.
* You start not from blank, slightly noisy input, but rather from an existing image - thus the resulting effects latch on to preexisting visual patterns, distorting elements
of the image in a somewhat artistic fashion.
* The input images are processed at different scales (called *octaves*), which improves the quality of the visualization.
#### Implementing DeepDream in Keras
You'll start from a convnet pretrained on ImageNet. In Keras, many such convnets are available: VGG16, VGG19, Xception, ResNet50, and so on. You can implement DeepDream
with any of them, but your convnet of choice will naturally affect your visualizations, because different convnet architectures result in different learned features. The 
convnet used in the original DeepDream release was an Inception model, and in practice Inception is known to produce nice-looking DeepDreams, so you'll use the Inception V3
model that comes with Keras.

Next, you'll compute the *loss:* the quantity you'll seek to maximize during the gradient-ascent process. In Deep Learning for Computer Vision, you tried to maximize the
value of a specific filter in a specific layer. Here, you'll simultaneously maximize the activation of all filters in a number of layers. Specifically, you'll maximize a
weighted sum of the L2 norm of the activations of a set of high-level layers.

Lower layers reult in geometric patterns, whereas higher layers result in visual in which you can recognize some classes from ImageNet. You'll start from a somewhat arbitrary
configuration involving four layers - but you'll definitely want to explore many different configurations later.

Finally: the actual DeepDream algorithm. First, you define a list of *scales* (also called *octaves*) at which to process the images. Each successive scale is large than
previous one by a factor of 1.4: you start by processing a small image and then increasingly scale it up.

For each successive scale, from the smallest to the largest, you run gradient ascent to maximize the loss you previously defined, at that scale. After each gradient ascent
run, you upscale the resulting image by 40%.
