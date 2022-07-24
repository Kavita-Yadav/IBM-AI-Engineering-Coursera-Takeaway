# Introduction to Image Classification:

1. whats is Image Classification?
2. Challenges of Image Classification.

#### What is Image Classification?
Image classification is the process of taking an image or picture and getting a computer to automatically classify it, or try providing the probability of the class of the image. A class is essentially a label, for instance: cat, car, building, etc.

*Uses of Image Classification:*
- Organize Photo albums on smart devices.
- Augment Medical Professionals:  Image classification is also used in radiology to help medical professionals identify anomalies in X-rays.
- Identify images around Self-sriving cars: Image classification is also used in self driving cars to classify images around them to help the car identify how to navigate the road. 

#### Challenges of Image Classification:
- Change in viewpoint, change of illumination,deformation,occlusion,background clutter.

#### Supervised machine learning methods for image classification:
- K-Nearest Neighbors
- Feature Extraction
- Linear Classifiers

*k-NN:*
- k-NN means k-Nearest Neighbor
- Simplest classification algorithm
- Uses the most common and nearest classes to find closes match.
- Model & Training & Testing:
    - Split dataset into Training/Testing sets. Smaller part for testing and larger part for training.
    - Build the model with a training set.
    - Use testing set to assess the performance or use the testing set for prediction.
    - When we have completed testing our model we should use all the data.
- Accuracy:
    - what is the accuracy of a classifier ?
    - Ans: The number of samples that have predicted correctly divided by the total number of samples. We use a subset called the validation data to determine the best k, this is called a hyper parameter. To select the Hyperparameter we split our data set into three parts, the training set, validation set and test set. We use the training set for different hyperparameters, we use the accuracy for K on the validation data. We select the hyperparameter K that maximizes accuracy on the validation set. We use the test data to see how the model will perform on the real world. We will combine the validation set and test set to make things simpler.
- KNN is not usually used in practice, knn is extremely slow, and it can’t deal with many of the Challenges of Image Classification

*Linear Classifiers:*
- Learnable Parameters: `z= wx+b` w= wight, b = bias. `z=w1x1+w2x2+b`
- Logistic regression: Logistic Function called sigmoid.
- If you have the learnable parameters, you can use a linear classifier, you take the photo. Under the hood, you app will use the linear classifier to make a prediction and output the class as a string.

*Logistic regression Training:Gradient Descent:*
- Architecture:
                Train
    Dataset -----------> Classifier
                            |
                        Unknown Sample
- Cost and Loss:  Training is where you find the best learnable parameters of the decision boundary.
    - Loss: A loss function tells you how good your prediction is. The following loss is called the classification loss. Each time our prediction is correct, the loss function will output a zero. Each time our prediction is incorrect, the loss function will output a one. 
    - Cost: It is sum of the loss. It tells us how good our learnable parameters are doing on the dataset. For each incorrectly classified samples, the loss is one, increasing the cost. Correctly classified samples do not change the cost. the cost is a function of multiple parameters w and b. We use the cross entropy loss that uses the output of the logistic function. The cross entropy deals with how likely the image belongs to a specific class. If the likelihood of belonging to an incorrect class is large, the cross entropy loss in turn will be large. If the likelihood of belonging to the correct class is correct, the cross entropy is small, but not zero.
- Gradient Descent: A method to find the best learnable parameters. If we find the minimum of the cost, we can find the best parameter. Generally, the more parameters you have, the more images and iterations you need to make the model work. We can choose a learning rate that's too small. We can choose a learning rate that's too large. With a good learning rate, we will reach the minimum of the cost

*Mini-Batch Gradient Descent:*
- Mini-Batch Gradient Descent, it will allow you to train models with more data. In this, we select batch. When we use all the samples in the dataset we call it an epoch.
- Iterations = training size/batch size.
- At the end of each epoch we calculate the accuracy on the validation data. We repeat the process for each iteration. If the accuracy decreases we have trained too much. This is called overfitting.

*SoftMax and Multi-Class Classification:*
- The argmax function returns the index corresponding to the largest value in a sequence of numbers.
- Method to convert two calss classifier into multi class classifier:
   * One-vs-rest
   * One-vs-one

*Support vector machine:*
- Kernels: A dataset is linearly separable if we can use a plane to separate each class, but not all data sets are linearly separable. Different type of kernel: Linear,Polynomial, Radial basis function(RBF). RBF is most popular. where the classifier fits the data points not the actual pattern, this is a case of overfitting, where higher gamma the more likely we will over fit.
- Find best gamma value using validation data. Split the data into training and validation set. Use the validation samples to find the Hyperparameters.
- SVMs are based on the idea of finding a plane that best divides a dataset into two classes. . So, the goal is to choose a hyperplane with as big a margin as possible. We try to find the hyperplane in such a way that it has the maximum distance to support vectors.

*Image Features:*
- H.O.G stands for Histogram of oriented Gradients. The technique counts occurrences of gradient orientation in localized portions of an image. HOG would generate a Histogram for each of these regions separately. The histograms are created using the gradients and orientations of the pixel values, hence the name ‘Histogram of Oriented Gradients’. The HOG feature vector is a combination of all pixel-level histograms and used with SVM to classify the image. this example is simplified, we must also consider other free parameters like number of image cells or how many angle bins in the histogram.
- There are other types of features for images like SURF and SIFT.
- Feature extraction, Kernel i.e non-linear mapping, Linear classification.

NoteL Question 5
You train a Support Vector Machine and obtain an accuracy of 100% on the training data and 50% on the validation data. This is an example of overfitting.