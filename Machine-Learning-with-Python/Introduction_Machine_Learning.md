What is machine Learning?

Machine learning is the subfield of computer science that gives "computers the ability to learn without being explicity programmed".
Machine Learning is allow to make model that train machine to identify all possible feature of object untill it able to recongnise it correctly.

Real Time Machine Learning examples:
1. Netflix recommendation of TV shows, Movies.
2. Loan application approval system in banks. They use machine learning to predict the probability of default for each applicant, and then approve or
3. refuse loan application based on that probability.
4. Telecommunication companies use their customers demographic data to segment them, or predict if they will unsubscribe from their company the next month.
5. Other examples are chatbots, logging into our phones or even computer games using face recognition.

Major Machine Learning Technique:
- Regression/Estimation: Predicting continuos values.
- Classification: Predicting the item class/category of a case.
- Clustering: Finding the structure of data; summarization
- Associations: Associating frequent co-occuring items/events.
- Anomaly Detection: Discovering abnormal and unusual cases.
- Sequence mining: Predicting next events; click-stream(Markov Model,HMM)
- Dimension Reduction: Recducing the size of data (PCA)
- Recommendation systems: Recommending items

Difference between artificial intelligence, machine learning, and deep learning:
- AI Components: try to make an computer intelligent in order to mimic the cognitve functions of humans.
  - Commputer Vision
  - Language Processing
  - Creativity
  - Etc.
- Machine Learning:It is the branch of AI that covers the statistical part of aritifical intelligence.
It teaches the computer to solve problems by looking at hundreds of thousands of exampkes, learning
from them, and then using that experience to solve the same problem in new situations.
  - Classification
  - Clustering
  - Neural Network
  - Etc.
- Revolution in ML:
  - Deep learning: It involves a deeper level of automation in comparision with most machine learning
  algorithms.
  
  There are two important components of machine learning which this course will cover:
  - Machine learning applications: what is the purpose of machine learning and where it can be applied in real world.
  - Machine learning algorithms: you'll get a general overview of Machine Learning topics, such as supervised or 
  unsupervised learning, model evaluation and various Machine Learning algorithms.
  
  Python libraries that can use for machine learning:
  - NumPy
  - SciPy
  - matplotlib
  - pandas
  - scikit learn: 
    - It is free software machine learning library.
    - Classification, Regression and Clustering algorithms.
    - Works with NumPy and SciPy.
    - Great Documentation.
    - Easy to implement. Most of the task is already included in this library.
    Data Preprocessing-> Train/Test split -> Algorithm setup -> Model fitting -> Prediction -> Evaluation -> Model export
    - scikit-learn functions:
      - fix data using preprocessing. Preprocessing package of SciKit Learn provides several common utility functions and transformer classes to change
      raw feature vectors into a suitable form of vector for modeling.
      ```
      from sklearn import preprocessing
      X = preprocessing.StandardScaler().fit(X).transform(X)
      ```
      - Split dataset into train and test sets to train your model and then test the model's accuracy separately.
      ```
      from skelarn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
      ```
      - Set up an algorithm. Example: build a classifier using a suport vector classification algorithm. 
      [//]: # CLF = estimator instance. Let's Initialize its parameters. C parameter in SVM is Penalty parameter of the error term. You can consider it as the degree of correct classification that the algorithm has to meet or the degree of optimization the the SVM has to meet. For greater values of C, there is no way that SVM optimizer can misclassify any single point.
      ```
      from sklearn import svm
      clf = svm.SVC(gamma= 0.001, C=100.)
      ```
      - train the model with the train set by passing our training set to the fit method, CLF model learns to classify unknown cases.
      ```
      clf.fit(X_train, y_train)
      ```
      - now test set to run predictions and then result tell us what the class of each unknown value is.
      ```
      clf.predict(X_test)
      ```
      - can use the different metrices to evaluate model accuracy. Eg: using a confusion matrix to show results.
      ```
      from sklearn.metrics import confusion_matrix
      print(confusion_matrix(y_test, yhat, labels=[1,0]))
      ```
      - finally, save model
      ```
      import pickle
      s= pickle.dumps(clf)
      ```
      
Supervised vs Unsupervised Learning:  

- What is supervised learning?
Supervise means to observe and direct the execution of task,project or activity.
Here we are going to supervise machine learning model.
- How to supervise a machine learning model ?
We "teach the model" then with that knowledge it can predict unknown or future instances. 
- How do we teach a model?
We teach a model with labeled dataset.
- What is labeled dataset?
 Labeled data is a group of samples that have been tagged with one or more lables.
 Labelling typically takes a set of unlabeled data and augments each piece of it with informative tags.
- what is the difference between labeled and unlabeled dataset?
Labeled data is data that comes with a tag, like a name, a type, or a number. Unlabeled data is data that comes with no tag. Columns of data are called features.

```
For example: we have data of Pets.
Pets labelled data can have label of : name of each pets(eg: dog, cat, rat), breed of different pets, also the weight and height can also be included.
But in unlabbled data that information will not be available.
```
There are two types of data commonly used in machine learning i.e numeric and categorical(non-numeric). Categorical dataset is made for classification.

Come back to supervised learning, we have two types of supervised learning techniques which are classification and regression.
- Classification is the process of predicting a discrete class label or category.
- Regression is the process of predicting continous values as opposed to predicting a categorical value in classification.
      
- what is unsupervised learning?
The model works on its own to discover information. Unsupervised learning techniquesare Dimension reduction, Density estimation, Market basket analysis and Clustering.
  - Dimension reduction: Dimensionality reduction or feature selection play a large role in this by reducing redundant features to make the classification easier.
  - Market basket analysis is modeling technique based upon the theory that if you buy a certain group of items, you're more likely to buy another group of items.
  - Density estimation is very simple concept that is mostly used to explore the data to find some structure within it.
  - Clustering is considered to be one of the most popular unsupervised machine learning techniques used for grouping data points, or objects that are somehow similar. It has manu applications: Discovering structure, Summarization, Anomaly detection.

Difference between Supervised vs Unsupervised learning:

| Supervised Learning | Unsupervised Learning |
| --- | --- |
| Classification: Classifies labled data | Clustering: Finds patterns and groupings from unlabeled data. |
| Regression: Predicts trends using previous labled data | Has fewes evaluation methods than supervised learning |
| Has more evaluation methods than unsupervised learning | less controlled environment |
| Controlled environment |  |

    
    
