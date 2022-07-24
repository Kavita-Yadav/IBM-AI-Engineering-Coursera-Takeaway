#### Introduction to Tensor Flow:
- It is open source software library by Google.
- Intially created for heavy numerical computational tasks
- Main Application: Machine Learning & Deep Neural Networks
- C/C++ backend
- It used as structure known as `data flow graphs`.

#### Why should adapt TensorFlow ?
- Python and C++ API
- Faster compile times
- Supports CPUs, GPUs and distributed processing

#### Data Flow Graph:
- We crate a graph with the following computation units:
    - Nodes = Mathematial Operations.
    - Edges = Multi-Dimentional Array(Tensors)
- The standard usage: build >> execute.

#### What is tensor?
The data that’s passed between the operations are Tensors. In effect, a Tensor is a multidimensional array. 

#### Computation Graph ingredients:
```
tf.variable()---->
                  | --> tf.matmul() -> tf.add() -> Tensor as Result
tf.placeholder()->

|___________________________________________________________________|
                                |
    Add all these operations in a graph and create a tf.session()
```

#### Architecture of TensorFlow:
```
                Python front end
        Core TensorFLow Execution System
        CPU     GPU      Android     iOS
```

#### Why Deep Learning is suitable with tensorFLow?
- Extensive built-in support for deep learning.
- It has collection of athematical functions that is useful for neural networks.
- Auto-differentiation and first-rate optimizers.

#### TensorFlow 2.x:
- In TF 2.x, Keras is the default high-level API for TensorFlow.
    * Keras is known for ease-of-use but does not have its own execution engine.
    * Tight integration between Tensorflow and Keras allows building powerful deep learning models more quickly and easily.
- TF 2.x includes performance optimization & GPU enhancements.
- TF 2.x supports Eager Execution mode and is activated by default for TF low-level API.
- What is Eager execution Mode?
```
# Without eager execution
import tensorflow as tf
import numpy as np
a= tf.constant(np.array([1.,2.,3.]))
b=tf.costant(np.array([4.,5.,6.]))
c=tf.tensordot(a,b,1)

session=tf.Session()
output=session.run(c)
session.close()

# With eager execution, without changing the code. Just change the datatype of tensor from 
# tensorflow.python.framework.ops.Tensor -> tensorflow.python.framework.ops.EagerTensor
# It will help to obtain intermediate results anywhere in the code.
import tensorflow as tf
import numpy as np
a= tf.constant(np.array([1.,2.,3.]))
b=tf.costant(np.array([4.,5.,6.]))
c=tf.tensordot(a,b,1)
output=c.numpy()
```

#### Why deep learning?
- Cancer detection
- Drug discovery
- Image classification
- Speech recognition
- Video captioning
- Real-time translation
- Lane tracking
- Vehicle detection
- Face recognition
- Video surveillance

#### What is deep learning?
- Deep Learning
    - Supervised, semi-supervised and unsupervised methods
- Deep Neural Network
    - Neural Networks with more than 2 layers
    - Sophisticated mathematical modeling
    - To extract feature sets automatically.
        * images, videos, sound and text.


#### Deep Neural Networks:
- Convolution Neural Networks (CNNs)
- Recurrent Neural Networks  (RNNs)
- Restricted Boltzmann Machines (RBMs)
- Deep Belief Networks (DBNs)
- Autoencoders

*Convolution Neural Networks (CNNs):*
- Traditional Approach:
1. Time consuming.
2. Requires domain experts.
3. Not scalable
```
Dataset -> Feature Extraction -> Shallow Neural Network -> Output
```
- Deep Learning Approach:
1. More accurate
2. Easily to scale
```
Dataset -> Feature Extraction -> Shallow Neural Network -> Output
```
Convolutional Neural Network is a deep learning approach that learns directly from samples in a way that is much more effective than traditional Neural networks. CNNs achieve this type of automatic feature selection and classification through multiple specific layers of sophisticated mathematical operations. Through multiple layers, a CNN learns multiple levels of feature sets at different levels of abstraction. And this leads to very effective classification. 

- CNN Application:
1. Object Detection in Images eg: self-driving cars
2. Coloring blacn and white images and creating art images.

*Recurrent Neural Networks(RNNs):*
- A Recurrent Neural Network, or RNN for short, is a type of deep learning approach, that tries to solve the problem of modeling sequential data. Whenever the points in a dataset are dependent on the previous points, the data is said to be sequential. For example, a stock market price is a sequential type of data because the price of any given stock in tomorrow’s market, to a great extent, depends on its price today. As such, predicting the stock price tomorrow, is something that RNNs can be used for. We simply need to feed the network with the sequential data, it then maintains the context of the data and thus, learns the patterns within the data. 
- It can also use for sentiment analysis. For example, you’re scrolling through your product catalogue on a social network site and you see many comments related to a particular product of yours. Rather than reading through dozens and dozens of comments yourself and having to manually calculate if they were mostly positive, you can let an RNN do that for you. Indeed, an RNN can examine the sentiment of keywords in those reviews. Please remember, though, that the sequence of the words or sentences, as well as the context in which they are used, is very important as well. By feeding a sentence into an RNN, it takes all of this into account and determines if it the sentiment within it those product reviews are positive or negative. 
- RNNs can also be used to predict the next word in a sentence. I’m sure we’ve all seen how our mobile phone suggests words when we’re typing an email or a text. It is language modeling where the model has learned from a big textual corpus.
- Language translation by google translator. It uses a probability model that has been trained on lots of data where the exact same text is translated into another language. 
- Speech-to-text. The recognized voice is not only based on the word sound; RNNs also use the context around that sound to accurately recognize of the words being spoken into the device’s microphone. 

*Restricted Boltzmann Machines (RBMs):*
- Restricted Boltzman Machines, or RBMs, are used to find the patterns in data in an unsupervised manner. They are shallow neural nets that learn to reconstruct data by themselves. They are very important models, because they can automatically extract meaningful features from a given input, without the need to label them. 
- They are building blocks of other networks eg: DBN.
- RBM useful for:
1. Feature Extraction
2. Dimensionality reduction
3. Pattern Recognition
4. Recommender System
5. Handling Missing valyes
6. Topic Modeling.

*Deep Belief Networks (DBNs):*
- A Deep Belief Network is a network that was invented to solve an old problem in traditional artificial neural networks. Which problem? The back-propagation problem, that can often cause “local minima” or “vanishing gradients” issues in the learning process. A DBN is built to solve this by the stacking of multiple RBMs.
- Application of DBNs:
1. Used for classification:Image Recognition
2. Very accurate discriminative classifier: Using stack of RBMs.

*Autoencoder:*
- Autoencoders were invented to address the issue of extracting desirable features. 
- Autoencoders try to recreate a given input, but do so with a slightly different network architecture and learning method. Autoencoders take a set of unlabeled inputs, encodes them into short codes, and then uses those to reconstruct the original image, while extracting the most valuable information from the data. 
- Applications:
1. Unsupervised tasks: Dimensionality reduction task, Feature extraction, Image recognition.
For example, to detect a face in an image, the network encodes the primitive features, like the edges of a face. Then, the first layer's output goes to the second Autoencoder, to encode the less local features, like the nose, and so on. Therefore, it can be used for Feature Extraction and image recognition. 