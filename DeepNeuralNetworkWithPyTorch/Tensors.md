# Overview of Tensors:
- Tensors 1-D and 2-D.
- Derivatives.

### Neural Networks and Tensors:

Data -> Dataset ->Tensor -> Input -> Hidden layer -> Parameters

For example if we want to use a database as an input for a neural network, we could do so in the following manner. In this example each row of the database can be treated as a Pytorch tensor and we input each of these tensors into a neural network. Thus we see that a tensor is simply just a vector or a rectangular array consisting of numbers. We can also easily integrate Pytorch with "GPU". This is an important factor for training neural networks. Parameters in neural networks are a kind of tensor that allow you to calculate gradients or derivatives. Gradients and derivatives will allow you to train the neural network.

#### Tensor 1-D:
Tensors are arrays that are the building blocks of a neural network. A 0-d tensor is just a number, 1-D tensor is an array of numbers. It could be: A row in a data database, A vector, Time series.

- Types: A tensor contains elements of a single data type. The tensor type is the type of tensor. When we are dealing with real numbers, the tensor type could either be a float tensor or a double tensor. When we are dealing with unsigned integers that are used in 8 bit images, the tensor type is a byte tensor. Thus, we see we have a variety of different tensor types depending upon the data type of the elements in the tensor.

```
# Create tensor
import torch
a=torch.tensor([7,4,3,2,6])

# find type of data
a.dtype

# find type of tensor
a.type()

# can specify the datatype of tensor
a=torch.tensor([7,4,3,2,6],dtype=torch.int32)

#create a tensor of specific type
a=torch.FloatTensor([0,1,2,3,4])

# convert the type of tensor
a=a.type(torch.FloatTensor)

# size of tensor
a.size()

# rank of the tensor
a.ndimension()

# convert 1-d tensor into 2-d
a_col = a.view(5,1)

# If dont know the no. of row and col use below method
a_col = a.viea(-1,1)

# convert numpy array to torch tensor
numpy array = np.array([0.0,1.0,2.0,3.0,4.0])
torch_tensor = torch.from_numpy(numpy_array)
```

- Indexing and Slicing
- Basic Operations
- Universal Functions


