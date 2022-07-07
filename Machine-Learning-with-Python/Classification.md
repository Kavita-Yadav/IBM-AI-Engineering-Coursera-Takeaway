# Introduction to Classification
 
 Classification determines the class label for an unlabeled test case.
 -  Decision Trees(ID3, C4.5, C5.0)
 - Naive Bayes
 - Linear Discriminant Analysis
 - K-Nearest Neighbor
 - Logistic Regression
 - Neural Networks
 - Support Vector Machines(SVM)
 
### K-Nearest Neighbours:
#### What is K-Nearest Neighbor (or KNN)?
  - A method for classifying cases based on their similarity to other cases.
  - Cases that are near each other are said to be "neighbors".
  - Based on similar cases with same class labels are near each other. 
  
#### The K-Nearest Neighbors algorithm
- Pick a value for K.
- Calculate the similarity/distance of unknown case from all cases.
- Select the K-observations in the training data that are "nearest" to the unknown data point.
- Predict the response of the unknown data point using the nmost popular response value from the K-nearest neighbors.
 
#### What is the best value of K for KNN?
Where the K-value gives the best accuracy. KNN can also be used for regression.

#### Evaluation metrics in classification
- Evalucation metrics are used to check the performance of model. These are the following evaluation metrics used in classification:
  - Jaccard index]
  - F1-score: Close to 1 is better.
  - Log loss: Performace of a classifier where the predicted output is a probability value between o and 1. Lower logg less has better accuracy.
