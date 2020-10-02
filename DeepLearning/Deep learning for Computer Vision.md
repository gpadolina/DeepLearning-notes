## Deep learning for computer vision
This chapter introduces convolutional neural networks, also known as *convnets*, a type of deep-learning model almost universally used in computer vision applications.
## Introduction to convnents
A convnet takes as input tensors of shape (image_height, image_width, image_channels), not including the batch dimension.
#### The convolution operation
The fundamental difference betwen a densely connected layer and a convolution layer is this: *Dense* layers learn global patterns in their input feature space, whereas
convolution layers learn local patterns: in the case of images, patterns found in small 2D windows of the inputs.

This key characteristics gives convnets two interesting properties:
* *The patterns they learn are translation invariant.* After learning a certain pattern in the lower-right corner of a picture, a convnet can recorgnize it anywhere. A
densely connected network would have to learn the pattern anew if it appeared at a new location. This make convnets data efficient when processing images because the
*visual world is fundamentally translation invariant):* they need fewer training samples to learn representations that have generalization power.
* *They can learn spatial hierarchies of patterns.* A first convolution layer will learn small local patterns such as edges, a second convolutional layer will learn patterns
made of the features of the first layers, and so on. This allows convnets to efficiently learn increasingly complex and abstract visual concepts.

Convolutions operate over 3D tensors, called *feature maps,* with two spatial axes (*height* and *width*) as well as a *depth* axis (also called the *channels* axis). For an
RBG image, the dimension of the depth axis is 3, because the image has three color channels: red, green, and blue. For a black-and-white picture, the depth is 1 levels of gray.

Convolutoins are define by two key parameters:
* *Size of the patches extracted from the inputs* - These are typically 3 x 3 or 5 x 5.
* *Depth of the output feature map* - The number of filters computed by the convolution.

In Keras Conv2D layers, these parameters are the first arguments passed to the layer: Conv2D(output_depth, (window_height, window_width)).

A convolution works by *sliding* these windows of size 3 x 3 or 5 x 5 over the 3D input feature map, stopping at every possible location, and extracting the 3D patch of
surrounding features. Each such 3D patch is then transformed via a tensor product with the same learned weight matrix, called the *convolution kernel* into a 1D vector of
shape.

Note that the output width and height may differ from the input width and height. They may differ for two reasons:
* Border effects, which can be countered by padding the input feature map
* The use of *strides*

If you want to get an output feature map with the same spatial dimensions as the input, you can use *padding.* Padding consists of adding an appropriate number of rows and
columns on each side of the input feature map so as to make it possible to fit center convolution windows around every input tile.

The other factor that can influence output size is the notion of *strides.* The description of convolution so far has assumed that the center tiles of the convolution
windows are all contiguous. But the distance between two successive windows is a parameter of the convolution, called its *stride*, which defaults to 1. It's possible to
have *strided convolutions:* convolutions with a stride higher than 1.

To downsample feature maps, instead of strides, we tend to use the *max-pooling* operation.
#### The max-pooling operation
The role of max pooling is to aggressively downsample feature maps, much like strided convolutions.

In short, the reason to use downsampling is to reduce the number of feature-map coefficients to process, as well as to induce spatial-filter hierarchies by making successive
convolution layers look at increasingly large windows in terms of the fraction of the original input they cover.

Note that the max pooling isn't the only way you can achieve such downsampling. As you already know, you can also use strides in the prior convolution layer. And you can use
average pooling instead of max pooling, where each local input patch is transformed by taking the average value of each channel over the patch, rather than the max. But max
pooling tends to work better than these alternative solutions. In a nutshell, the reason is that features tend to encode the spatial presence of some pattern or concept over
the different tiles of the feature map, and it's more informative to look at the *maximal presence* of different features than at their *average presence.*

## Training a convnet from scratch on a small dataset
As a practical example, we'll focus on classifying images as dogs or cats, in a dataset containing 4,000 pictures of cats and dogs (2,000 each). We'll use 2,000 pictures
for training - 1,000 for validation, and 1,000 for testing.

Then we'll introduce *data augmentation,* a powerful technique for mitigating overfitting in computer vision. By using data augmentation, you'll improve the netwrok to reach
an accuracy of 82%.

In the next section, two more essential techniques for applying deep learning to small datasets will be covered: *feature extraction with a pretrained network* (which will
get to an accuracy of 90% to 96%) and *fine-tuning a pretrained network* (this will get to a final accuracy of 97%).
#### The relevance of deep learning for small-data problems
One fundamental characteristics of deep learning is that it can find interesting features in the training data on its own, without any need for manual feature engineering,
and this can only achieved when lots of training examples are available.

But what constitutes lots of samples is relative - relative to the size and depth of the network you're trying to train, for starters. It isn't possible to train a convnet
to solve a complex problem with just a few tens of samples, but a few hundred can potentially suffice if the model is small and well regularized and the task is simple.
Because convnets learn local, translation-invariant features, they're highly data efficient on perceptual problems. Training a convnet from scratch on a very small image
dataset will still yield reasonable results despite a relative lack of data, without the need for any custom feature engineering.

What's more, deep-learning models are by nature highly repurposable: you can take say an image-classification or speech-to-text model trained on a large-scale dataset and
reuse it on a significantly different problem with only minor changes.
#### Building your network
You'll resue the same general structure: the convnet will be a stack of alternated Conv2D (with *relu* activation) and MaxPooling2D layers. But because you're dealing with
bigger images and a more complex problem, you'll make your network larger, accordingly. Because you're attacking a binary-classification problem, you'll end the network with
a single unit (a *Dense* layer of size 1) and a *sigmoid* activation. This unit will encode the probability that the network is looking at one class or the other.

For the compilation step, you'll go with the *RMSprop* optimizer, as usual. Because you ended the network with a single sigmoid unit, you'll use binary crossentropy as the
loss.
#### Data preprocessing
The steps for getting data formatted into appropriate format are as follows:
1. Read the picture files.
2. Decode the JPEG content to RGB grids of pixels.
3. Convert these into floating-point tensors.
4. Rescale the pixel values (between 0 and 255) to the [0, 1] interval.

It may seem a bit daunting, but fortunately Keras has utilities to take care of the steps automatically. Keras a module with image-processing helper tools, located at
*keras.preprocessing.image.* In particular, it contains the class ImageDataGenerator, which lets you quickly set up Python generators that can automatically turn image files
on disk into batches of preprocessed tensors.

Note that the generator yields these batches indefinitely: it loops endlessly over the images in the target folder. For this reason, you need to break the iteration loop at
some point.

Let's fit the model to the data using the generator. You do so using the *fit_generator* method, the equivalent of *fit* for data generators like this one. It expects as its
first argument a Python generator that will yield batches of inputs and targets indefinitely, like this one does. Because the data is being generated endlessly, the Keras
model needs to know how many samples to draw from the generator before declaring an epoch over. This is the role of the *steps_per_epoch* argument: after having drawn
*steps_per_epoch* batches from the generator - that is, after having f run for *steps_per_epoch* gradient descent steps - the fitting process will go to the next epoch.
#### Using data augmentation
Overfitting is caused by having too few samples to learn from, rendering you unable to train a model that can generalize to new data. Given infinite data, your model would
be exposed to every possible aspect of the data distribution at ahand: you would never overfit. Data augmentation takes the approach of generating more training data from
existing training samples, by *augmenting* the samples via a number of random transformations that yield believable-looking images.

If you train a new network using this data-augmentation configuration, the network will never see the same input twice. But the inputs it sees are still heavily
intercorrelated, because they come from a small number of original images - you can't produce new information, you can only remix existing information. As such, this may
not be enough to completely get rid of overfitting. To further fight overfitting, you'll also add a *Dropout* layer to your model, right before the densely connected
classifier.

By using regularization techniques even further, and by tuning the network's parameters such as the number of filters per convolution layer, or the number of layers in the
network, you may be able to get an even better accuracy, likely up to 87%. But it would prove difficult to go any higher just by training your own convnet from scratch,
because you have so little data to work with. As a next step to improve your accuracy on this problem, you'll have to use a pretrained model.

## Using a pretrained convnet
A common and highly effective approach to deep learning on small image datasets is to use a pretrained network. A *pretrained network* is a saved network that was previously
trained on a large dataset, typically on a large-sclae image-classification task. If this original dataset is large enough and general enough, then the spatial hierarchy of
features learned by the pretrained network can effectively act as a generic model of the visual world, and hence its features can prove useufl for many different computer
vision problems. Such portability of learned features across different problems is a key advantage of deep learning compared to many older, shallow-learning approaches, and
it makes deep learning very effective for small-data problems.

You'll use the VGG16 architecture; it's a simple and widely used convnet architecture for ImageNet. There are two ways to use a pretrained network: *feature extraction* and
*fine-tuning.*
#### Feature extraction
Feature extraction consists of using the representations learned by a previous network to extract interesting features from new samples. These features are then run through
a new classifier, which is trained from scratch.

As you previously, convnets used for image classfication comprise two parts: they start with a series of pooling and convolution layers, and they end with a densely connected
classifier. The first part is called the *convolutional base* of the model.

Why only reuse the convolutional base? Could you resue the densely connected classifier as well? In general, doing so should be avoided. The reason is that the represenations
learned by the convolutional base are likely to more generic and therefore more reusable: the feature maps of a convnet are presence maps of generic concepts over a picture,
which is likely to be useful regardless of the computer-vision problem at hand. For problems where object location matters, densely connected features are largely useless.

Layers that come earlier in the model extract local, highly generic feature maps such as visual edges, colors, and textures, whereas layers that are higher up extract more
abstract concepts.

Here's the list of image-classification models that are available as part of *keras.applications:*
* Xception
* Inception V3
* ResNet50
* VGG16
* VGG19
* MobileNet

#### Feature extraction with data augmentation
Let's review the second technique for doing feature extraction, which is much slower and more expensive, but which allows you to use data augmentation during training:
extending the *conv_base* model and running it end to end on the inputs.

Because models behave just like layers, you can add a model like *conv_base* to a *Sequential* model just like you would add a layer.

Before you compile and train the model, it's very important to freeze the convolutional base. *Freezing* a layer or set of layers means preventing their weights from being
updated during trained. If you don't do this, then the representations that were previously learned by the convolutional base will be modified during training. Because the
*Dense* layers on top are randomly initialized, very large weight updates would be propagated through the network, effectively destroying the representations previously
learned.

In Keras, you freeze a network by setting its *trainable* attribute to *False.*
#### Fine-tuning
Another widely used technique for model reuse, complementary to feature extraction, is *fine-tuning.* Fine-tuning consists of unfreezing a few of the top layers of a frozen
model base used for feature extraction, and jointly training both the newly added part of the model and these top layers. This is called *fine-tuning* because it slightly
adjusts the more abstract representations of the model being reused.

The steps for fine-tuning a network are as follows:
1. Add your custom network on top of an already-trained base network.
2. Freeze the base network.
3. Train the part you added.
4. Unfreeze some layers in the base network.
5. Jointly train both these layers and the part you added.

You'll fine-tune the last three convolutional layers, which means all layers up to *block4_pool* should be frozen, and the layers *block5_conv1, block5_conv2,* and
*block5_conv3* should be trainable.

Why not fine-tune more layers? Why not fine-tune the entire convolutional base? You could. But you need to consider the following:
* Earlier layers in the convolutional base encode more-generic, reusable features, whereas layers higher up encode more-specialized features. It's more useful to fine-tune
the more specialized features, because these are the ones that need to be repurposed on your new problem.
* The more parameters you're training, the more you're at risk at overfitting.

Now you can begin fine-tuning the network. You'll do this with the RMSprop optimizer, using a very low learning rate. The reason for using a low learning rate is that you
want to limit the magnitude of the modifications you make to the representations of the three layers you're fine-tuning.
#### Wrapping up
Here's what you should take away from the last two sections:
* Convnets are the best type of machine-learning models for computer-vision tasks. It's possible to train one from scratch even on a very small datasetm with decent results.
* On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when you're working image data.
* It's easy to reuse an existing convnet on a new dataset via feature extraction. This is a valuable technique for working with small image datasets.
* As a complement to feature extraction, you can use fine-tuning, which adats to a new problem some of the representations previously learned by an existing model.
## Visualizing what convnets learn
It's often said that deep-leanring models are "black boxes": learning representations that are difficult to extract and present in a human-readable form. Although this is
partially true for certain types of deep-learning models, it's definitely not true for convnets. The representations learned by convnets are highly amenable to visualization,
in large part because they're *representations of visual concepts.* Since 2013, a wide array of techniques have been developed for visualizing and interpreting these
representations.
* *Visualizing intermediate convnet outputs (intermediate activations)* - Useful for understanding how successive convnet layers transform their input, and for getting a
first idea of the meaning of individual convnet filters.
* *Visualizing convnets filters* - Useful for understanding precisely what visual pattern or concept each filters in a convnet is receptive to.
* *Visualizing heatmaps of class activation in an image* - Useful for understanding which parts of an image were identified as belonging to a given class, thus allowing
you to localize objects in images.

#### Visualizing intermediate activations
Visualizing intermediate activations consists of displaying the feature maps that are output by various convolution and pooling layers in a network, given a certain input
(the output of a layer is often called its *activation,* the output of the activation function).

In order to extract the feature maps you want to look at, you'll create a Keras model that takes batches of images as input, and outputs the activations of all convolution
and pooling layers. To do this, you'll use the Keras class *Model.* A model is instantiated using two arguments: an input tensor (or list of input tensors) and an output
tensor (or list of output tensors). What sets the *Model* class apart is that it allows for models with multiple outputs, unlike *Sequential.*

In the general case, a model can have any number of inputs and outputs. This one has one input and eight outputs: one output per layer activation.

There are a few things to note here:
* The first layer acts as a collection of various edge detectors. At that stage, the activations retain almost all of the information present in the initial picture.
* As you go higher, the activations become increasingly abstract and less visually interpretable. They begin to encode higher-level concepts such as "cat ear" and "cat eye."
Higher presentations carry increasingly less information about the visual contents of the image, and increasingly more information related to the class of the image.
* The sparsity of the activations increases with the depth of the layer: in the first layer, all filters are activated by the input image; but in the following layers, more
and more filters are blank.

A deep neural network effectivvely acts as an *information distillation pipeline,* with raw data going in (in this case, RGB pictures) and being repeatedly transformed so
that irrelevant information is filter out (for example, the specific visual appearance of the image), and useful information is magnified and refined.

This is analogous to the way humans and animals perceive the world. After observing a scene for a few seconds, a human can remember which abstract objects were present in
it (bicycle, tree) but can't remember the specific appearance of these objects. Your brain has learned to completely abstract its visual input - to transform it into
high-level visual concepts while filtering out irrelevant visual details - making it tremendously difficult to remember how things around you look.
