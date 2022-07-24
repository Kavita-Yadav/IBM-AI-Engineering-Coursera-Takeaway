# Object Detection:
1. Sliding Windows
2. Bounding Box
3. Bounding Box Pipeline
4. Score

*Sliding Windows:*
Sliding-window object detection is a popular technique for identifying and localizing objects in an image. The approach involves scanning the image with a ﬁxed-size rectangular window and applying a classiﬁer to the sub-image deﬁned by the window. For eg, if we want to detect car in an image using sliding windows approach then. The part of image which overlaps with the window is taken and fed into the car classifier to determine whether a car is in it or not. It will use car or no-car classifier.

*Bounding Box:*
In object detection, we usually use a bounding box to describe the target location. The bounding box is a rectangular box that can be determined by the x and y axis coordinates in the upper-left corner and the x and y axis coordinates in the lower-right corner of the rectangle.

*Bounding Box Pipeline:*
Object detection with superhuman ability Bounding box, polygon, point and line tools enable you to construct a predictable pipeline of high-quality training data that will teach your ML-powered computer vision system to find and identify objects in image and video data.

*Score:*
The score is a number between 0% and 100% that indicates confidence that the object was genuinely detected. The closer the number is to 1, the more confident the model is. You can decide a cut-off threshold below which you will discard detection results.

#### Object Detection with Haar Cascade Classifier:
- It is a machine learning method.
- Trained on both positive and negative images.
- Base on the Haar wavelet sequence.
- It use the integral image concept: Integral image is a concept that uses the cumulative sum of pixels above and to the left of the current pixel cell.
- An AdaBoost classifier is used to reduce the number of features:
    * A weak classifier is made on top of the training data based on the weighted samples
    * It selects only those features that help to  improve the classifier accuracy.
    * AdaBoost cuts down the number of features significantly.
- A cascade of classifiers:
                                         Yes
Input Image -> Sub Image -> Classifier 1 ---> Classifier 2
                                | No             | No
                                    Not an Object
