import numpy as np
import torch
import matplotlib.pyplot as plt


# generator for randomness for reproducability
g = torch.Generator().manual_seed(42)


def load_names():
  # Load data
  words = open('data/names.txt', 'r').read().splitlines()
  token = '.'
  # set will remove duplicates (characters)
  alphabet = sorted(list(set(''.join(words))))
  # 26 letters + start and end token
  num_tokens = len(alphabet) + 1
  # map characters to integers
  # 0 is reserved for the start and end token
  chr_to_int = {ch: i+1 for i, ch in enumerate(alphabet)}
  chr_to_int[token] = 0
  int_to_chr = {i: ch for ch, i in chr_to_int.items()}
  return words, token, num_tokens, chr_to_int, int_to_chr


def create_linear_data(dim=1, num_samples=1000, plot=False):
  """Create data for linear regression."""
  # create random data
  x = np.random.rand(num_samples, dim) # (samples, dim)
  w_true = np.random.rand(dim, 1) # (dim, 1)
  b_true = np.random.rand(1) # (1,)
  # f(x) = y = w * x + b
  y_true = (x @ w_true) + b_true # (samples, 1)
  # add noise
  noise = np.random.normal(loc=0, scale=0.01, size=y_true.shape) # (samples,)
  # noise = np.random.randn(*y_true.shape) # (samples,)
  y_true += noise # (samples, 1)
  # absorb bias into weights (with x0 = 1, b = w0)
  x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1) # (samples, dim + 1)
  w_true = np.concatenate([w_true, b_true.reshape(1, -1)], axis=0) # (dim + 1, 1)
  # check shapes
  # print(x.shape, w_true.shape, b_true.shape, y_true.shape)
  if dim == 1 and plot:
    # plot data
    plt.scatter(x[:, 0], y_true)
    plt.show()
  return x, y_true, w_true, b_true


def create_random_data(num_samples=1000, dim=1, binary=False, step_fct=False):
  """Create random data for classification or regression."""
  x = torch.rand(num_samples, dim)
  if binary:
    # pytorch needs 0 or 1 for binary classification
    y = torch.Tensor([1 if np.random.uniform(0, 1) > 0.5 else 0 for _ in range(num_samples)])
    # step function
    if step_fct:
      y = torch.Tensor([1 if torch.sum(xi) > 0.5 else 0 for xi in x])
  else:
    y = torch.rand(num_samples)
  return x, y


def train_test_split(X, y, test_size=0.2):
  """Split data into train and test sets."""
  # split data
  n = len(X)
  n_test = int(n * test_size)
  n_train = n - n_test
  X_train = X[:n_train]
  X_test = X[n_train:]
  y_train = y[:n_train]
  y_test = y[n_train:]
  # check shapes
  # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
  return X_train, X_test, y_train, y_test