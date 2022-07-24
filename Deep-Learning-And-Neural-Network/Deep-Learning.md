## Deep Learning Application:
- Color Restoration: Automatic colorization and color restoration in black and white images.
- Speech Reenactment: Synhing lip movement in a video with an audio clip.
- Automatic Handwriting Generation
- Automatic Machine Translation
- Automatically Adding Sounds to Silent Movies
- Object Classification and Detection in Images
- Self Driving Cars

### Artificial Neural Network:
- In Artificial Neural Network, there are three types of layer: Input layer, Output layer and Hidden layer(Between Input and Output layer).
- Three main topics associated with artificial neural networks:
    1. Forward Propagation.
    2. Backpropagation.
    3. Activation Functions.

#### Forward Propagation: 
Forward propagation is the process through which data passes through layers of neurons in a neural network from the input layer all the way to the output layer. A neural network without an activation function is essentially just a linear regression model.

#### Gradient Descent:
Optimize z value.

#### Back Propagation:
1. Calculate the error between the ground truth and the estimated output. Let's denote the error by E.
2. Propagate the error back into the network and update each weight and bias as per the equations.
3. Complete Training Algorithm:
- Intialize the weights and the biases.
- Iteratively repeat the following steps:
    1. Calculate network output using forward propagration.
    2. Calculate error between ground truth and estimated or predicted output.
    3. Update weights and biases through backpropagation.
    4. Repeat the above three steps until number of iterations/epochs is reached or error between ground truth and predicted output is below a predefined threshold.

#### Vanishing Gradient:
Gradients tend to get smaller and smaller as we keep on moving backward in the network. This means that the neurons in the earlier layers learn very slowly as compared to the neurons in the later layers in the network. The earlier layers in the network, are the slowest to train. The result is a training process that takes too long and a prediction accuracy that is compromised. Accordingly, this is the reason why we do not use the sigmoid function or similar functions as activation functions, since they are prone to the vanishing gradient problem. 

#### Activation Functions:
There are seven types of activation function:
- Binary Step Function
- Linear Function
- Sigmoid Function
- Hyperbolic Tangent Function
- ReLU(Rectified Linear Unit)
- Leaky ReLU
- Softmax Function

*Sigmoid Function:*
The function is pretty flat beyond the +3 and -3 region. This means that once the function falls in that region, the gradients become very small. This results in the vanishing gradient problem that we discussed, and as the gradients approach 0, the network doesn't really learn. Another problem with the sigmoid function is that the values only range from 0 to 1. This means that the sigmoid function is not symmetric around the origin. The values received are all positive. Well, not all the times would we desire that values going to the next neuron be all of the same sign. This can be addressed by scaling the sigmoid function, and this brings us to the next activation function: the hyperbolic tangent function. 

*Hyperbolic tangent Function:*
It is actually just a scaled version of the sigmoid function, but unlike the sigmoid function, it's symmetric over the origin. It ranges from -1 to +1. However, although it overcomes the lack of symmetry of the sigmoid function, it also leads to the vanishing gradient problem in very deep neural networks.

*ReLU(Rectified Linear Unit):*
The rectified linear unit, or ReLU, function is the most widely used activation function when designing networks today. In addition to it being nonlinear, the main advantage of using the ReLU, function over the other activation functions is that it does not activate all the neurons at the same time. It makes the network sparse and very efficient. Also, the ReLU function was one of the main advancements in the field of deep learning that led to overcoming the vanishing gradient problem.

*Softmax Function:*
The softmax function is ideally used in the output layer of the classifier where we are actually trying to get the probabilities to define the class of each input. So, if a network with 3 neurons in the output layer outputs [1.6, 0.55, 0.98] then with a softmax activation function, the outputs get converted to [0.51, 0.18, 0.31]. This way, it is easier for us to classify a given data point and determine to which category it belongs.

*Conclusion:*
The sigmoid and the tanh functions are avoided in many applications nowadays since they can lead to the vanishing gradient problem. The ReLU function is the function that's widely used nowadays, and it's important to note that it is only used in the hidden layers. Finally, when building a model, you can begin with using the ReLU function and then you can switch to other activation functions if the ReLU function does not yield a good performance.

### Deep Learning Libraries:
- TensorFlow
- Keras
- PyTorch
- Theano ( no longer maintained)

*Keras vs PyTorch vs TensorFlow:*
- TensorFlow is the most popular deep learning library, developed by Google.
- PyTorch is a cousin of the Lua-based Torch framework, and is a strong competitor to TensorFlow.
- Keras is the easiest API to use and the go-to library for quick prototyping and fast development.

#### Regression Models with Keras:
Steps to build a model:
1. Get Dataset. 
2. Build a network that takes eight inputs or predictors, consists of two hidden layers, each of five neurons, and an output layer. Next, let's divide our dataset into predictors or target.
2. Identify the predictors or Target. Eg: concrete dataset has 9 column name as: `Cement, Blast Furnace Slag, Fly Ash, Water, SUperplasticizer, Coarse Aggregate, Fine Aggregate, Age, Strength`. So there will be 8 column that are going to informative columns(`Cement, Blast Furnace Slag, Fly Ash, Water, SUperplasticizer, Coarse Aggregate, Fine Aggregate, Age`) which are all ingredients of concrete used to help determine the Stength of concrete. So the target/predictor column will be `Strength`.
3. After determining predictors/target columns. Let's call the model eg: 
```
from keras.model import Sequential
model = Sequential()
```
4. Now let's build a layers:
```
## Add hidden layers to model; 
from keras.layers import Dense
## the number of columns or predictors in our dataset
n_cols = concrete_data.shape[1]
## 1st input layer with neurons and input_shape; 5 is the number of neurons in each dense layer
model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
## 2nd hidden layer
model.add(Dense(5, activation='relu'))
## output layer
model.add(Dense(1))
```
5. For training need to define an optimizer and the error matrics in our compiler:
```
## The "adam" optimizer is that you don't need to specify the learning rate which we need to do in gradient descent
model.compile(optimizer='adam',loss='mean_squared_error')
```
6. Use the fit method to train our model:
```
model.fit(predictors, target)
```
7. Once training is complete, we can start making predictions using the predict method:
```
model.predict(test_data)
```

#### Classification Models with Keras:
Steps to build a model:
1. Get Dataset. In below Eg used car dataset. The decision is 0, meaning that buying this car would be a bad choice. A decision of 1 means that buying the car is acceptable, a decision of 2 means that buying the car would be a good decision, and a decision of 3 means that buying the car would be a very good decision.
2. Use `one-hot encoding` to change label of any categorical column from string to numerical. 
3. Buils a network that takes eight inputs or predictors, consists of two hidden layers, each of five neurons, and an output layer. Next, let's divide our dataset into predictors and target.
4. Identify the predictors and Target. Eg: car dataset has 9 column name as: `price_high, price_low, price_med, maintenance_high, maintenance_low, maintenance_med, persons_2, persons_more`. So there will be 8 column that are going to predictors columns(`decision`) which have all information help to determine the best decision to buy good car. So the target column will be `decision`. decision= 0: Not acceptable, 1: Acceptable, 2:Good, 3: Very Good.
5. For classification problems, we can't use the target column as is; we actually need to transform the column into an array with binary values similar to one-hot encoding.
```
## Using the "to_categorical" function from the Keras utilities package.
target = to_categorical(car_data['decision'])
```
6. After determining predictors and target columns. Let's call the model eg: 
```
from keras.model import Sequential
model = Sequential()
```
4. Now let's build a layers:
```
## Add hidden layers to model; 
from keras.layers import Dense
## the number of columns or predictors in our dataset
n_cols = concrete_data.shape[1]
##  The additional import statement here is the "to_categorical" function in order to transform our target column into an array of binary numbers for classification.
target = to_categorical(target)
## 1st input layer with neurons and input_shape; 5 is the number of neurons in each dense layer
model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
## 2nd hidden layer
model.add(Dense(5, activation='relu'))
## Output layer with 4 neurons(because 4 type of o/p expected in decision column)
## Also specify the softmax function as the activation function for the o/p layer, so that the sum of the predicted values from all the neurons in the o/p layer sum nicely to 1.
model.add(Dense(4), activation='softmax')
```
5. For training need to define an optimizer and the error matrics in our compiler:
```
## The "adam" optimizer is that you don't need to specify the learning rate which we need to do in gradient descent
## "accuracy" is a built-in evaluation metric in Keras. You can also define your own accuracy if you want and then can pass it as parameter below.
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
```
6. Use the fit method to train our model:
```
## Number of epochs can also be defines in regression model if you want too.
model.fit(predictors, target, epochs=10)
```
7. Once training is complete, we can start making predictions using the predict method:
```
model.predict(test_data)
```

### Shallow Versus Deep Neural Networks:
- A shallow neural network consists of one hidden layer.
- A deep neural network consists of more hidden layers and has a large number of neurons in each layer.
- But why did deep learning take off just recently?

### Why deep learning took off?
1. Advancement in the field itself. ReLU activation function is one of the functionused for deep learning neural networks.
2. Data. -> Deep neural networks work best when trained with large and large amounts of data, since neural networks learn the training data so well, then large amounts of data have to be used in order to avoid overfitting of the training data. 
3. The other conventional machine learning algorithms, while they do improve with more data, but up to a certain point. After that, no significant improvement would be observed with more data. In case of deep learning; the more data you feed it, the better it performs.
4. Computational Power. With NVIDIA's super powerful GPUs, we are now able to train very deep neural networks on tremendous amount of data in a matter of hours as opposed to days or weeks, which is how long it used to take to train very deep neural networks.

### Supervised Learning Models:
- Convolutional Neural Network.
- Recurrent Neural Network.

#### Convolutional Neural Network:
1. Convolutional Neural Networks(CNNs) are similar to the typical neural networks.
2. CNNs take input as images.
3. This allows us to incorporate properties that make the training process much more efficient.
4. Solve problems involving image recongnition, object detection and other computer vision applications.

*CNN Architecture:*
```
Input Image -> Convolution layer -> Pooling layer -> Convolution layer -> Pooling Layer -> Fully Connected Layer(with specific number of neurons acc to data) -> output(with specific number of neurons acc to data).
        |________________|__________________|                  |
            Convolution  |_____________________________________|
                                Max Pooling
```

*Input Layer:*
- The input to a convolutional neural network, is mostly an (n x m x 1) for grayscale images or an (n x m x 3) for colored images, where the number 3 represents the red, green, and blue components of each pixel in the image.

*Convolutional Layer:*
- In the convolutional layer, we basically define filters and we compute the convolution between the defined filters and each of the three images(taking the eg of red, blue, green images). The more filters we use, the more we are able to preserve the spatial dimensions better. 
*Note:* Why would we need to use convolution? Why not flatten the input image into an (n x m) x 1 vector and use that as our input? Well, if we do that, we will end up with a massive number of parameters that will need to be optimized, and it will be super computationally expensive. Also, decreasing the number of parameters would definitely help in preventing the model from overfitting the training data. It is worth mentioning that a convolutional layer also consists of ReLU's which filter the output of the convolutional step passing only positive values and turning any negative values to 0.

*Pooling Layer:*
- The pooling layer's main objective is to reduce the spatial dimensions of the data propagating through the network. There are two types of pooling that are widely used in convolutional neural networks. Max- pooling and average pooling. In max-pooling which is the most common of the two, for each section of the image we scan we keep the highest value.
- Similarly, with average pooling, we compute the average of each area we scan. In addition to reducing the dimension of the data, pooling, or max pooling in particular, provides spatial variance which enables the neural network to recognize objects in an image even if the object does not exactly resemble the original object.

*Fully Connected Layer:*
- The fully connected layer, we flatten the output of the last convolutional layer and connect every node of the current layer with every other node of the next layer. This layer basically takes as input the output from the preceding layer, whether it is a convolutional layer, ReLU, or pooling layer, and outputs an n-dimensional vector, where n is the number of classes pertaining to the problem at hand.

*Keras Code for convolutional network building:*
1. Build a model:
```
model = Sequential()
## Give image size as input
input_shape = (128,128,3)
```
2. Add Layers:
```
# A 1st convolutional layer, with 16 filters, each filter being of size 2x2 and slides through the image with a stride of magnitude 1 in the horizontal direction, and of magnitude 1 in the vertical direction.
model.add(Conv2D(16, kernel_size=(2,2), strides=(1,1),activation='relu', input_shape=input_shape))
# A 1st pooling layer, using max-pooling here with a filter or pooling size of 2 and the filter slides through the image with a stride of magnitude 2.
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# A 2nd convolutional layer, with 32 filters, each filter being of size 2x2 .
model.add(Conv2D(32, kernel_size=(2,2), activation='relu', input_shape=input_shape))
# A 2nd pooling layer, using max-pooling here with a filter or pooling size of 2.
model.add(MaxPooling2D(pool_size=(2,2)))
```
3. Flatten output from above layer so that the data can proceed to the fully connected layers.
```
model.add(Flatten())
```
4. Add another dense layer with 100 nodes and output layer has number of nodes equal to the number of classes.
```
model.add(Dense(100,activation='relu'))
# We use the softmax activation function in order to convert the outputs into probabilities
model.add(Dense(num_classes, activation='softmax'))
```

#### Recurrent Neural Networks:
Recurrent neural networks or (RNNs) for short, are networks with loops that don't just take a new input at a time, but also take in as input the output from the previous dat point that was fed into the network.
Recurrent neural networks are very good at modelling patterns and sequences of data, such as texts, genomes, handwriting, and stock markets.

*LSTM:*
A very popular type of recurrent neural network is the long short-term memory model or the (LSTM) model for short. It has been successfully used for many applications including image generation, where a model trained on many images is used to generate new novel images. Another application is handwriting generation. Also LSTM models have been successfully used to build algorithms that can automatically describe images as well as streams of videos.

### Unsupervised Learning Models:
- Autoencoder

*Autoencoder:*
- Autoencoding is a data compression algorithm where the compression and the decompression functions are learned automatically from data. instead of being engineered by a human.
- Autoencoders are data-specific, which means that they will only be able to compress data similar to what they have been trained on. Therefore, an autoencoder trained on pictures of cars would do a rather poor job of compressing pictures of buildings, because the features it would learn would be vehicle or car specific.
- Interesting applications of autoencoders are data denoising and dimensionality reduction for data visualization. Here is the architecture of an autoencoder.
- *Algorithm:*
```
Input image -> Encoder -> Compressed Representation -> Decoder -> Output Image
- Autoencoder is an unsupervised neural network model.
- It tries to predict x from x without need for any lables.
- Autoencoders can learn data projections that are more interesting than PCA or other basic techniques.
```
- A very popular type of autoencoders is the Restricted Boltzmann Machines(RBMs).
- Application include:
    1. Fixing Imbalanced Datasets.
    2. Estimating Missing Values.
    3. Automatic Feature Extraction.



