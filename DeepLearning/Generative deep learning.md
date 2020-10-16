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

To avoid losing a lot of image detail after each successive scale-up (resulting in increasingly blurry or pixelated images), you can use a simple trick: after each scale-up,
you'll reinject the lost details back into the image, which is possible because you know what the original image should look like at the larger scale. Given a small image
size S and a large image size L, you can compute the difference between the original image resized to size L and the original resized to size S - this difference quantifies
the details lost when going from S to L.

Layers that are lower in the network contain more-local, less-abstract representations and lead to dream patterns that look more geometric. Layers that are higher up lead
to more-recognizable visual patterns based on the most common objects found in ImageNet, such as dog eyes, bird feathers, and so on.
#### Wrapping up
* DeepDream consists of running a convnet in reverse to generate inputs based on the representations learned by the network.
* The results produced are fun and somewhat similar to the visual artifacts induced in humans by the disruption of the visual cortex via psychedelics.
* Note that the process isn't specific to image models or even to convnets. It can be done for speech, music, and more.
## Neural style transfer
In addition to DeepDream, another major development in deep-learning-driven image modification is *neural style transfer,* introduced by Leon Gatys et al. in the summer of
2015. The neural style transfer algorithm has undergone many refinements and spawned many variations since its original introduction, and it has made its way into many
smartphone photo apps.

Neural style transfer consists of applying the style of a reference image to a target image while conserving the content of the target image. In this context, *style*
essentially means textures, colors, and visual patterns in the image, at various spatial scales; and the *context* is the higher-level macrostructure of the image.

The key notion behind implementing style transfer is the same idea that's central to all deep-learning algorithms: you define a loss function to specify what you want to
achieve, and you minimize this loss. You kno what you want to achieve: conserving the content of the original image while adopting the style of the reference image. If we
were able to mathematically define *content* and *style,* then an appropriate loss function to minimize would be the following.

#### The content loss
As you already know, activations from earlier layers in a network contain *local* information about the image, whereas activations from higher layers contain increasingly
*global, abstract* information.

A good candidate for content loss is thus the L2 norm between the activations of an upper layer in a pretrained convnent, computed over the target image, and the activatoins
of the same layer computed over the generated image.
#### The style loss
The content loss only uses a single upper layer, but the style loss as defined by Gatys uses multiple layers of a covnent: you try to capture the appearance of the style
reference image at all spatial scales extracted by the convnet, not just a single scale. For the style loss, Gatys use the *Gram matrix* of a layer's activations: the inner
produce of the feature maps of a given layer. This inner product can be understood as representing a map of the correlations between the layer's features.

In short, you can use a pretrained convnet to define a loss that will do the following:
* Preserve content by maintaining similar high-level layer activations between the targent content image and the generated image. The convnet should "see" both the target
image and the generated image as containing the same things.
* Preserve style by maintaning similar *correlations* within activations for both low-level layers and high-level layers. Feature correlations capture *textures:* the
generated image and the style-reference image should share the same textures at different spatial scales.
#### Neural style transfer in Keras
Neural style transfer can be implemented using any pretrained convnet. Here, you'll use the VGG19 network used by Gatys. VGG19 is a simple variant of the VGG16 network,
with three more convolutional layers.

This is the general process:
1. Set up a network that computes VGG19 layer activations for the style-reference image, the target image, and the generated image at the same time.
2. Use the layer activations computed over these three images to define the loss function described ealier, which you'll minimize in order to achieve style transfer.
3. Set up a gradient-descent process to minimize this loss function.

Let's set up the VGG19 network. It takes as input a batch of three images: the style-reference image, the target image, and a placeholder that will contain the generated
image. A placeholder is a symbolic tensor, the values of which are provided externally via Numpy arrays.

In the original Gatys paper, optimization is performed using the L-BFGS algorithm, so that's what you'll use. This is a key difference from the DeepDream example. The
L-BFGS algorithm comes packaged with SciPy, but there are two slight limitations with the SciPy implementation:
* It requires that you pass the value of the loss function and the value of the gradients as two two separate functions.
* It can only be applied to flat vectors, whereas you have a 3D image array.

Finally, you can run the gradient-ascent process using SciPy's L-BFGS algorith, saving the current generated image at each iteration of the algorithm (here, a single
iteration represents 20 steps of gradient ascent).

Keep in mind that what this technique achieves is merely a form of image retexturing, or texture transfer. It works best with style-reference images that are strongly
textured and highly self-similar, and with content targets that don't require high levels of detail in order to be recognizable. It typically can't achieve fairly abstract
feats such as tranferring the style of one portrait to another. The algorithm is close to classical signal processing than to AI, so don't expect it to work like magic.
#### Wrapping up
* Style transfer consists of creating a new image that preserves the contents of a target image while also capturing the style of a reference image.
* Content can be captured by the high-level activations of a convnet.
* Style can be captured by the internal correlations of the activations of different layers of a convnet.
* Hence, deep learning allows style transfer to be formulated as an optimization process using a loss defined with a pretrained convnet.
* Starting from this basic idea, many variants and refinements are possible.
## Generating images with variational autoencoders
Sampling from a latent space of images to create entirely new images or edit existing ones is currently the most popular and successful application of creative AI. In this
section and the next, we'll review some high-level concepts pertaining to image generation, alongside implementations details relative to the two main techniques in this
domain: *variational autoencoders* (VAEs) and *generative adversarial networks* (GANs).
#### Sampling from latent spaces of images
The key idea of image generation is to develop a low-dimensional *latent space* of representations (which naturally is a vector space) where any point can be mapped to a
realistic-looking image. The module capable of realizing this mapping, taking as input a latent point and outputting an image (a grid of pixels), is called a *generator*
(in the case of GANs) or a *decoder* (in the case of VAEs).

GANs and VAEs are two different strategies for learning such talent spaces of image representations, each with its own characteristics. VAEs are great for learning latent
spaces that are well structured, where specific directions encode a meaningful axis of variation in the data. GANs generate images that can potentially be highly realistic,
but the latent space they come from may not have as much structure and continuity.
#### Variational autoencoders
Variational autoencoders, simultaneously discovered by Kingma and Welling in December 2013 and Rezende, Mohamed, and Wierstra in January 2014, are a kind of generative model
that's especially appropriate for the task of image editing via concept vectors. They're a modern take on autoencoders - a type of network that aims to encode an input to
a low-dimensional latent space and then decode it back - that mixes ideas from deep learning with Bayesian inference.

In practice, such classical autoencoders don't lead to particularly useful or nicely structured latent spaces. They're not much good at compression, either. For these reasons,
they have largely fallen out of fashion. VAEs, however, augment autoencoders with a little bit of statistical magic that forces them to learn continuous, highly structured
latent spaces. They have turned out to be a powerful tool for image generation.

A VAE, instead of compressing its input image into a fixed code in the latent space, turns the image into the parameters of a statistical distribution: a mean and a variance.
Essentially, this means you're assuming the input image has been generated by a statistical process, and that the randomness of this process should be taken into accounting
during encoding and decoding.
