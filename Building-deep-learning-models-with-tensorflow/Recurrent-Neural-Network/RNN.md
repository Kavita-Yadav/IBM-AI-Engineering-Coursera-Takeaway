## The Sequential Problem:
- The points in a dataset are dependent on the other points, the data is said to be sequential. A common example of this is a time series, such as a stock price, or sensor data, where each data point represents an observation at a certain point in time. There are other examples of sequential data, like sentences, gene sequences, and weather data.
- RNN can handle sequential dataset whereas tradition NN can't. For eg: If something happen unusaul it will store that information also.

## RNN:
```
        Data in -> RNN -> Data out
                    |
        Recurs new state to itself
```

Application:
1. Speech recognition
2. Image captioning
3. Sentiment analysis

## Recurrent Neural Network Problems:
- Must remember all states at any given time
    * Computationally expensive
    * only store states within a time window.
- Sensitive to changes in their parameters.
- Vanishing Gradient
- Exploding Gradient:  where the gradient grows exponentially off to infinity. Due to that model's capacity to learn will be diminished.

#### The Long Short Term Memory(LSTM) Model:
- Maintaining states in expensive
- Vanishing Gradient
- Exploding Gradient
- Solution: Long Short Term Memory 

LSTM maintains a strong gradient over many time steps. This means you can train an LSTM with relatively long sequences. An LSTM unit in Recurrent Neural Networks is composed of four main elements: the memory cell and three logistic gates. The memory cell is responsible for holding data. The write, read, and forget gates define the flow of data inside the LSTM. The write gate is responsible for writing data into the memory cell. The read gate reads data from the memory cell and sends that data back to the recurrent network, and the forget gate, maintains or deletes data from the information cell, or in other words determines how much old information to forget. 

- original LSTM has only 1 hidden layer.
```
lstm_cell =tf.contrib.rnn.BasicLSTMCell(hidden_size)
```
- Stacked LSTM: Output of the first layer to be fed to second layer of LSTM. The output of the first layer will feed as the input to the second layer. Then, the second LSTM blends it with its own internal state to produce an output.
```
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell]*2)
output,state =tf.nn.dynamic_rnn(stacked_lstm, input_data)
```

- Training LSTM

#### Language Modelling:
Language Modeling is the process of assigning probabilities to sequences of words. Language Modeling is a gateway into many exciting deep learning applications like Speech Recognition, Machine Translation, and Image Captioning. For example, a language model could analyze a sequence of words and predict which word is most likely to follow.

Input words -> LSTM 1 -> LSTM 2 -> LSTM Output -> Softmax layer -> Logit(Output)