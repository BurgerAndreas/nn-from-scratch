import numpy as np
import torch
import matplotlib.pyplot as plt


# generator for randomness for reproducability
g = torch.Generator().manual_seed(42)


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
  """
  Split data into train and test sets.
  For regression data or names.
  """
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


def load_names():
  """
  For names.
  """
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


def build_char_dataset(words, block_size, chr_to_int):
  """
  Like a list of bigrams, but with more characters.
  For names.
  """
  # context length: how many characters do we take to predict the next one? 
  x, y = [], []
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = chr_to_int[ch]
      x.append(context)
      y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append
  x = torch.tensor(x)
  y = torch.tensor(y)
  return x, y


def load_text(src='data/tiny_shakespeare.txt'):
  """
  For tiny shakespeare.
  """
  # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  with open(src, 'r', encoding='utf-8') as f:
    text = f.read()
  # here are all the unique characters that occur in this text
  vocabulary = sorted(list(set(text)))
  vocab_size = len(vocabulary)
  # create a mapping from characters to integers
  chr_to_int = { ch:i for i,ch in enumerate(vocabulary) }
  int_to_chr = { i:ch for i,ch in enumerate(vocabulary) }
  encode = lambda s: [chr_to_int[c] for c in s] # encoder: take a string, output a list of integers
  decode = lambda l: ''.join([int_to_chr[i] for i in l]) # decoder: take a list of integers, output a string
  return text, vocabulary, vocab_size, encode, decode


def train_val_split_text(data, train_size=0.9):
  """
  For tiny shakespeare.
  """
  # Train and test splits
  n = int(train_size*len(data)) 
  # first train_size % will be train, rest val
  train_data = data[:n]
  val_data = data[n:]
  return train_data, val_data


  
def plot_loss(losses, n_points=-1):
  # plt.plot(losses_log)
  if n_points == -1:
    loss_smooth = torch.tensor(losses)
  else:
    # average into n_points
    loss_smooth = torch.tensor(losses).view(-1, int(len(losses)/n_points)).mean(1)
  plt.plot(loss_smooth)
  plt.xlabel('iteration')
  plt.ylabel('loss')
  plt.title('Training loss')
  plt.show()
  return loss_smooth