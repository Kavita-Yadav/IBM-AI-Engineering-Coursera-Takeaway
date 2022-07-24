### CNN:
The CNN is a set of layers with each of them being responsible to detect a set of feature sets, and these features are going to be more abstract as it goes further into examining the next layer.
- It is better than Shallow Neural Networks. Because Shallow Neural Network's best feature selection process is hard. Also, extending the features to other types of images is not possible.
Keyfeature:
- Detect and classify objects into categories.
- Independence from pose,scale,illumination,conformation and clutter

Input Image -> Primitive Feature-> Object Parts-> Object

#### Applications:
- Signal and image processing.
- Handwritten text/digits recognition
- Natural object classification(photos and videos)
- Segmentation
- Face detection
- Recommender systems
- Speech recognition
- Natural Language Processing

#### CNN for Classification:

- Digit recognition problem:
1. Data pre-processing
2. Training
3. Inference

CNN layer -> Pooling layer -> Fully connected layer

- Convolution is a function that can detect edges, orientations, and small patterns in an image. 
- (28*28 Matrix)Image * ([5*5]*8 Matrix)Kernel = ([28*28]*8 Matrix0 Image
- intialize kernel with random value.
- Add relu activation function

| Concolution layer: Intialize Kernel | Convolution layer: Activation function | Max Pooling layer | 1024 vector Fully connected layer |
| Extraction & Feature learning part |  Extraction & Feature learning part |  Extraction & Feature learning part | Classifiaction|
|-- | -- | -- | -- |
| (28*28 Matrix)Image * ([5*5]*8 Matrix)Kernel = ([28*28]*8 Matrix)Image |  ([28*28]*8 Matrix)Image | [14*14]*8 Matrix  | Input from previous pooling layer matrix -> Flatten -> Fully connected nodes -> relu activation -> read out -> softmax -> output
```

* CNN Architecture:*
![CNN_ARch.PNG](https://github.com/Kavita-Yadav/IBM-AI-Engineering-Coursera-Takeaway/blob/main/Building-deep-learning-models-with-tensorflow/Convolutional-Neural-Network/CNN_ARch.PNG "CNN Architecture")