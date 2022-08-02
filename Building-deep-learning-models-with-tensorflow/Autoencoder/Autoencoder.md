## Autoencoders:

- It is unsupervised type of machine learning.
- It will find the pattern in a dataset by detecting key features.
- Applications:
    * Autoencoders are used for tasks that involve:
        - Feature extraction
        - Data compression
        - Learning generative models of data
        - Dimensionality reduction
- Autoencoder is not the only dimension reduction method in Machine Leaning. Principal Component Analysis (or PCA) has been around for a long time, and is a classic algorithm for dimensionality reduction. 
- Autoencoder vs RBM: See picture
- Autoencoder Architecture: See picture
- Learening process of autoencoder:
```
Input -> Autoencoder -> Reconstructed Input
    |_____________________|
             Loss
```

```
Loss = tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(Loss)
```