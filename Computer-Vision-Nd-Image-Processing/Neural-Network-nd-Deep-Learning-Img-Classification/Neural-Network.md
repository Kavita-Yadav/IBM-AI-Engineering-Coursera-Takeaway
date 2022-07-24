#### Fully Connected Neural Network Architecture:

Note: Please refer to this course for more information about Neural Network "Introduction with Deep Learning & Neural Network with Keras".

input layer-> hidden layer 1-> hidden layer 2-> output layer

- If we have more than one hidden layer the neural network is called a deep neural network.
- More neurons or more layers may lead to overfitting 
- To perform gradient descent to obtain our learning parameters, we have to calculate the gradient, but the deeper the network, the smaller the gradient gets this is called the vanishing gradient.
- Dropout layers: helps with overfitting.
- batch normalization improvise training.
-  skip connections allow you too train deeper Networks by connecting deeper layers during training The hidden Layers of Neural networks replace the Kernels is SVM’s. We can use the raw Image or features like HOG Training.
- Neural networks are trainined in a similar manner to logistic regression and Softmax.
- The Loss or cost surface is complicated making training difficult.

#### Convolutional Networks:
- Hoe CNN's Build Features.
- Adding Layers.
- Receptive Field.
- Pooling
- Flattening and Fully connected layers.

*Architecture:*

Input -> Convolution + ReLU -> Pooling -> Convolution + ReLU -> Pooling-> Fully connected layers -> Classification/output

- Convolution and pooling layers are the first layers used to extract features from an input these can be thought of as the feature learning layers.
- Fully connected layers are simply a neural network.
- Both are learned simultaneously by minimizing the cross-entropy loss

*CNN Build Feature:*

*Adding layers:*

*Receptive Field:*
- Receptive Field is the size of the region in the input that produces a pixel value in the activation Map.

*Pooling:*
- MaxPooling is most popular. It helps to reduce the number of parameters of an input image and still preserves the important features.

*Flattening and Fully connected layers:*
- We simply flatten or reshape the output of the (Feature Learning layers)pooling layers and use them as an input to the fully connected layers.

#### CNN Architecture:
- Important CNN architectures include:
    - LeNet-5: The most successful use case of the LeNet-5 is the MNIST Dataset of handwritten digits.
    - AlexNet: ImageNet is a benchmark dataset i.e this is the one that everyone uses to see who has the best image classification method. But it has more parameters and more parameters need more data.
    - VGGNet: The VGG Network is a Very Deep Convolutional Network that was developed out of the need to reduce the number of parameters in the Convolution layers and improve on training time.
    - ResNet: ResNet help solve the problem by introducing Residual learning: Residual layers or skip connections allows the gradient to bypass different layers, improving performance. we can now build much deeper Networks.
- Transfer Learning; Transfer learning is where you use a pre-trained CNN to classify an image Instead of building your own network. Pre-trained CNN's have been trained with vast amount of data.