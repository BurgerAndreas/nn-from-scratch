import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.get_data import load_names, train_test_split


# https://www.youtube.com/watch?v=TcH_1BHy58I
# https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf


class CharMLP:
  """character bigram model. 
  Predicts the next character in a word based on only the previous character.
  Model works by counting occurances of bigrams.
  Not a neural network."""

  def __init__(self, block_size=3, dim_emb=10, num_neurons_hidden=50, epochs=10000):
    """Initializes the model, hyperparameters, and data."""
    # hyperparameters
    self.block_size = block_size
    self.dim_emb = dim_emb
    self.num_neurons_hidden = num_neurons_hidden
    # load data
    words, self.token, self.num_tokens, self.chr_to_int, self.int_to_chr = load_names()
    x, y = self.build_dataset(words)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # build model
    self.parameters = self.init_model()
    self.c, self.w1, self.b1, self.w2, self.b2 = self.parameters
    # find optimal learning rate
    lr_opt = self.find_lr(x_train, y_train, epochs=epochs, plot=True)
    # train model
    # self.train_torch(x_train, y_train, epochs=epochs, lr=0.0001)
    self.train_explicit(x_train, y_train, epochs=epochs, lr=0.001)
    # generate samples
    self.generate()
    # visualize embeddings
    # self.visualize_embeddings()

  def build_dataset(self, words):
    """
    Like a list of bigrams, but with more characters."""
    # context length: how many characters do we take to predict the next one? 
    x, y = [], []
    for w in words:
      context = [0] * self.block_size
      for ch in w + '.':
        ix = self.chr_to_int[ch]
        x.append(context)
        y.append(ix)
        #print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x, y
  
  def init_model(self):
    """Builds the model."""
    # input layer = embedding layer / lookup table for characters
    c = torch.randn((self.num_tokens, self.dim_emb), dtype=torch.float32, requires_grad=True)
    # hidden layer
    w1 = torch.randn((self.block_size * self.dim_emb, self.num_neurons_hidden), dtype=torch.float32, requires_grad=True)
    b1 = torch.randn((self.num_neurons_hidden,), dtype=torch.float32, requires_grad=True)
    # output layer
    w2 = torch.randn((self.num_neurons_hidden, self.num_tokens), dtype=torch.float32, requires_grad=True)
    b2 = torch.randn((self.num_tokens,), dtype=torch.float32, requires_grad=True)
    num_params = sum(p.numel() for p in [c, w1, b1, w2, b2])
    print(f'Number of parameters: {num_params}')
    return c, w1, b1, w2, b2

  def forward(self, x):
    """Forward pass."""
    # input layer
    x = self.c[x]
    # hidden layer
    x = x.view(-1, self.block_size * self.dim_emb)
    x = torch.matmul(x, self.w1) + self.b1
    x = F.relu(x)
    # output layer
    x = torch.matmul(x, self.w2) + self.b2
    return x

  
  def train_torch(self, x_train, y_train, batch_size=32, epochs=1000, lr=0.1):
    """Trains the model."""
    optimizer = torch.optim.SGD(self.parameters, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(epochs):
      optimizer.zero_grad()
      batch = torch.randint(0, x_train.shape[0], (batch_size,))
      y_hat = self.forward(x_train[batch])
      loss = loss_fn(y_hat, y_train[batch])
      loss.backward()
      optimizer.step()
      losses.append(loss.item())
      if epoch % (epochs/20) == 0:
        print(f'Epoch {epoch}: {loss.item()}')
    plt.plot(losses)
    plt.show()
    return losses
  

  def train_explicit(self, x_train, y_train, batch_size=32, epochs=1000, lr=0.1):
    """Trains the model."""
    losses = []
    for i in range(epochs):
      # minibatch construct
      batch = torch.randint(0, x_train.shape[0], (batch_size,))
      # forward pass
      emb = self.c[x_train[batch]] # (batch_size, block_size, dim_emb)
      h = torch.tanh(emb.view(-1, self.block_size*self.dim_emb) @ self.w1 + self.b1) # (batch_size, num_neurons_hidden)
      logits = h @ self.w2 + self.b2 # (batch_size, num_tokens)
      # loss
      loss = F.cross_entropy(logits, y_train[batch])
      # same as
      # counts = logits.exp()
      # prob = counts / counts.sum(dim=1, keepdim=True)
      # loss = -torch.log(prob[torch.arange(len(ytr)), ytr]).mean()
      # backward pass
      for p in self.parameters:
        p.grad = None
      loss.backward()
      # update
      for p in self.parameters:
        p.data += -lr * p.grad
      # track stats
      if i % (epochs/20) == 0:
        print(f'Epoch {i}: {loss.item()}')
      losses.append(loss.log10().item())
    return losses
  

  def find_lr(self, x_train, y_train, batch_size=32, epochs=100000, plot=False):
    """Find the optimal learning rate.
    By testing an exponentially distributed range of learning rates for one epoch each."""
    lrei = torch.linspace(-3, 0, epochs) # exponent for learning rate at each step
    lri = 10**lrei # learning rate at each step
    lossi = []
    for i in range(epochs):
      # minibatch construct
      batch = torch.randint(0, x_train.shape[0], (batch_size,))
      # forward pass
      emb = self.c[x_train[batch]] # (batch_size, block_size, dim_emb)
      h = torch.tanh(emb.view(-1, self.block_size*self.dim_emb) @ self.w1 + self.b1) # (batch_size, num_neurons_hidden)
      logits = h @ self.w2 + self.b2 # (batch_size, num_tokens)
      # loss
      loss = F.cross_entropy(logits, y_train[batch])
      # backward pass
      for p in self.parameters:
        p.grad = None
      loss.backward()
      # update
      lr = lri[i]
      for p in self.parameters:
        p.data += -lr * p.grad
      # track stats
      # if i % (epochs/20) == 0:
      #   print(f'Epoch {i}: {loss.item()}')
      lossi.append(loss.item())
    # plot
    if plot:
      plt.plot(lrei, lossi)
      plt.title('Optimal learning rate exponent at valley')
      plt.xlabel('log10(lr) = exponent of 10')
      plt.ylabel('loss')
      plt.show()
    # smooth loss with moving average of window size 10
    # lossi = np.convolve(lossi, np.ones(10)/10, mode='same')
    lossi = pd.Series(lossi).rolling(10).mean().values
    opt_lr = lri[np.argmin(np.array(lossi))]
    print(f'Optimal learning rate: {opt_lr:.3e}')
    return opt_lr
  

  def generate(self, num_samples=20):
    # sample from the model
    samples = []
    for _ in range(num_samples):
      out = []
      context = [0] * self.block_size # initialize with all ...
      while True:
        emb = self.c[torch.tensor([context])] # (1, block_size, d)
        h = torch.tanh(emb.view(1, -1) @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        # stop when .
        if ix == 0:
          break
      samples.append(''.join(self.int_to_chr[i] for i in out))
    print('Samples: \n', *samples)
    return samples


  def visualize_embedding(self):
    # visualize dimensions 0 and 1 of the embedding matrix C for all characters
    plt.figure(figsize=(8,8))
    plt.scatter(self.c[:,0].data, self.c[:,1].data, s=200)
    for i in range(self.c.shape[0]):
        plt.text(self.c[i,0].item(), self.c[i,1].item(), self.int_to_chr[i], ha="center", va="center", color='white')
    plt.grid('minor')
    plt.show()