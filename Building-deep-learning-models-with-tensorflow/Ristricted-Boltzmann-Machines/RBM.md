# Restricted Boltzmann Machines:

Part of colaborative Filtering


1. Input Layer(visible layer) -> Hidden Layer
2. Hidden Layer -> Input Layer

Imagine that we have a type of neural network that has only two layers, the input layer and the hidden layer. Let's also assume that this network has learned in such a way that it can reconstruct the input vectors. For example, when you feed the first user vector into the network, it goes through the network and finally fires up some units in the hidden layer. Then the values of the hidden layer will be fed back into the network and a vector which is almost the same as the input vector is reconstructed as output. We can think of it as making guesses about the input data. You feed the second user's ratings, which are not very different from the first user and thus the same hidden units will be turned on and the network output would be the same as the first reconstructed vector, so on.

- RBMs are shallow neural netowrk
- They have 2 layers
- They are unsuprvised
- Find patterns in data by reconstructing the input
- Application:
1. Dimensionality reduction
2. Feature Extraction
3. Collaborative filtering
4. Main component of DBM

*Learning Process of RBMs:* RBMs learn patterns and extract important features in data by reconstructing the input. So, the learning earning process consists of several forward and backward passes where the RBM tries to reconstruct the input data. The weights of the neural net are adjusted in such a way that the RBM can find the relationships among input features and then determines which features irrelevant. After training is complete, the net is able to reconstruct the input based on what it learned.

*RBMs training process:*
1. Forward Pass: In the forward pass, the input image is converted to binary values and then the vector input is fed into the network where its values are multiplied by weights and an overall bias in each hidden unit. Then result go to activation function such as sigmoid. That define which neuron may or may not activate.  So, as you can see the forward pass translates the inputs into a set of binary values that get represented in the hidden layer.
2. Backward Pass: In the backward pass, the activated neurons in the hidden layer send the results back to the visible layer, where the input will be reconstructed. During this step, the data that is passed backwards is also combined with the same weights and overall bias that were used in the forward pass. It gives the shape of the probability distribution of the input values given the hidden values. And sampling the distribution, the input is reconstructed. So, as you can see the backward pass is about making guesses about the probability distribution of the original input.
3. Quality Assesment: Assessing the quality of the reconstruction by comparing it to the original data. The RBM then calculates the error and adjust the weights and bias in order to minimize it. That is in each epoch, we compute the error as a sum of the squared difference between step 1 and the next step. These three steps are repeated until the error is deemed sufficiently low.

*Advantages of RBM:*
- RBMs are goot at handling unlabeled data like videos,photos and audio files, etc
- RBMs extract important feature from the input
- RBMs are more efficient at dimensionality reduction than PCA
- RBMs learn from the data. They actually encode their own structure. This is why they're grouped into a larger family of models known as autoencoders. However, restricted boltzmann machines differ from autoencoders and that they use a stochastic approach rather than a deterministic approach. 
