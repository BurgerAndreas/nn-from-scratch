import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from helpers.get_data import create_linear_data, train_test_split, create_random_data

# https://www.youtube.com/watch?v=VMj-3S1tku0&t=7146s
# https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb
# https://github.com/karpathy/micrograd

class Neuron:
  def __init__(self, num_in, print_shapes=False):
    #
    self.print_shapes = print_shapes
    # torch gives us (1) tensor operations and 
    # (2) automatic differentiation (backpropagation)
    # compared to using a list of floats
    # it's best to initialize weights close to 0 and all the same
    # or even better see kaiming initialization
    self.w = torch.rand(size=[num_in, 1], requires_grad=True) # (dim, 1)
    self.b = torch.rand(size=[1], requires_grad=True) # (1,)
  
  def __call__(self, x):
    """Return prediction of neuron on input x."""
    if self.print_shapes: print('  Neuron x:  ', *x.shape)
    out = torch.sigmoid(x @ self.w + self.b) # (samples, 1)
    if self.print_shapes: print('  Neuron out:', *out.shape)
    return out

  def __repr__(self):
    """Print parameters of neuron."""
    return f'Neuron({self.w}, {self.b})'
  
  def parameters(self):
    """Return parameters of neuron."""
    return [self.w, self.b]
  

class Layer:
  def __init__(self, num_in, num_out):
    """Basically a list of neurons."""
    # num_in = number of inputs for each neuron
    # = number of neurons in previous layer
    # num_out = number of neurons in layer
    self.neurons = [Neuron(num_in) for _ in range(num_out)]
  
  def __call__(self, x):
    """Return output of layer on input x."""
    # out_neuron = (samples, num_in)
    # out_layer = (samples, num_out) = (samples, num_neurons)
    return torch.cat([n(x) for n in self.neurons], dim=1)

  def __repr__(self):
    return f'Layer({len(self.neurons)} neurons, dim={self.neurons[0].w.shape[0]})'
  
  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]


class MLP:
  def __init__(self, num_in, num_outs, print_shapes=False):
    """A list of layers."""
    self.print_shapes = print_shapes
    # number of inputs = number of neurons in first layer
    # number of outputs = number of neurons in next layer
    n = [num_in] + num_outs
    self.layers = [Layer(n[i], n[i+1]) for i in range(len(num_outs))]
    # loss function
    self.loss_fct = torch.nn.BCELoss()
    # self.loss_fct = torch.nn.BCEWithLogitsLoss()
    
  
  def __call__(self, x):
    if self.print_shapes: print('MLP x:    ', x.shape)
    for layer in self.layers:
      x = layer(x)
      if self.print_shapes: print(' Layer out:  ', x.shape)
    return x

  def __repr__(self):
    return f'MLP({len(self.layers)} Layers: {self.layers})'
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

  def train(self, x_train, y_train, learning_rate=0.1, epochs=100):
    losses = []
    for k in range(epochs):
      # forward pass
      y_pred = self(x_train) # (samples, classes)
      y_train = torch.reshape(input=y_train, shape=y_pred.shape) # (samples, classes)

      # mean squared error loss
      # loss = torch.mean((y_train - y_pred)**2)
      # loss = torch.nn.functional.mse_loss(input=y_train, target=y_pred)

      # binary cross entropy loss
      # This will produce a nan gradient. Use BCELoss instead.
      # loss = torch.nn.functional.binary_cross_entropy(input=y_pred, target=y_train) # BCELoss
      # Combines a Sigmoid layer and the BCELoss
      # loss = torch.nn.functional.binary_cross_entropy_with_logits(input=y_train, target=y_pred) # BCEWithLogitsLoss

      loss = self.loss_fct(input=y_pred, target=y_train)

      # backward pass
      loss.backward()
      # update parameters
      for p in self.parameters():
        p.data -= learning_rate * p.grad
        p.grad.zero_()
      print(f'Epoch {k}: {loss.item():.3f}')
      losses.append(loss.item())
    return losses


def scratch_mlp(learning_rate=5., epochs=1000):
  """Train a multi-layer perceptron model."""
  # create data
  dim = 3
  x, y = create_random_data(num_samples=1000, dim=dim, binary=True, step_fct=True) # (samples, dim), (samples,)
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
  # plt.scatter(x=X_train, y=y_train, color='blue', label='True')
  # plt.title('Train data')
  # plt.show()
  # create model
  model = MLP(dim, [5, 2, 1])
  print('MLP:', model)
  # train model
  print('Training data:', X_train.shape, y_train.shape)
  losses = model.train(X_train, y_train, learning_rate=learning_rate, epochs=epochs)
  print(f'Initial loss: {losses[0]:.3f} Final loss: {losses[-1]:.3f}')
  # test model
  y_pred = model(X_test)
  # plot results
  if dim == 1:
    plt.scatter(x=X_test, y=y_test, color='blue', label='True')
    plt.scatter(x=X_test, y=y_pred.detach().numpy(), color='orange', label='Predicted')
    plt.legend()
    plt.title('Test and prediction data')
    plt.show()
  # plot loss
  plt.plot(losses)
  plt.title('Loss over Epochs')
  plt.show()
