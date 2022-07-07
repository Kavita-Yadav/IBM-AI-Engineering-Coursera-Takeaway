# Introduction to Classification

#### What is classification?
- A supervised learning approach.
- Categorizing some unknown items into a discrete set of categories or "classes".
- The target attribute is a categorical variable.

#### How does classofication work? 
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
  - Jaccard index.
  - F1-score: Close to 1 is better.
  - Log loss: Performace of a classifier where the predicted output is a probability value between o and 1. Lower logg less has better accuracy.

### Decision Tree
A Decision Tree is a type of clustering  approach that can predict the class of a group, for example, DrugA or DrugB.

** Decision tree learning algorithm **
- Choose an attribute from your dataset.
- Calculate the significance of attribute in splitting of data. 
- Split data based on the value of the best attribute.
- Go to step 1.

#### Building decision trees
 
 - What attribute is the best? Ans: More Predictiveness, Less Impurity, Lower Entropy.
 ** Entropy: ** Measure of randomness or uncertainity.The lower the Entropy, the less uniform the distribution, the purer the node. 
The entropy in a node is the amount of information disorder calculated in each node.
 - The tree with the higher Information Gain after splitting. Information gain is the information that can increase the level of certainity after splitting.

### Logistic Regression
Logistic regression is a classification algorithm for categorical variables.

Application of logiastic regression:
- Predicting the probability of a person having a heart attack.
- Predicting the mortality in injured patients.
- Predicting a customer's propensity to purchase a product or halt a subscription.
- Predicting the probability of failure of a given process or product.
- Predicting the likelihood of a homeowner defaulting on a mortgage.

When is logistic regression suitable?
- If your data is binary: 0/1, Yes/No, True/False.
- If you need probabilistic results.
- When you need a linear decision boundary.
- Logistic Regression can be used to understand the impact of a feature on a dependent variable.Logistic regression is analogous to linear regression but takes a categorical/discrete target field instead of a numeric one.Logistic Regression measures the probability of a case belonging to a specific class.
- Sigmoid function/logistic function 

Logistic Regression Training:
- Change the parameters of model for optimization.
- change the weight -> reduce the cost.
- cost function or MSE.

Minimizing the cost function of the model:
- How to find the best parameters for our model ? Ans: Minimize the cost function.
- How to minimize the cost function? Ans: using Gradient Descent.
- What is gradient descent? Ans: A technique to use the derivative of a cost function to change the parameter values, in order to minimize the cost.
- Training algorithm recap:
  - Initalize the parameters randomly.
  - Feed the cost function with training set, and calculate the error.
  - Calculate the gradient of cost function.
  - Update weights with new values.
  - Go to step 2 until cost is small enough.
  - Predict the new customer X.
 
 ### Support Vector Machine
 SVM is a supervised algorithm that classifies cases by finding a separator.
 1. Mapping data to a high dimensional feature space.
 2. Finding a separator.  

Advantages:
- Accurate in high-dimensional spaces.
- Memory efficient
Disadvantages:
- Prone to over-fitting
- No probability estimation
- Small datasets

SVM application:
- Image recognition
- Text category assignment
- Detecting spam
- Sentiment analysis
- Gene Expression Classification
