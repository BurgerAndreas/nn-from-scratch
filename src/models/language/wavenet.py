import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math as m

from helpers.get_data import load_names, train_test_split, build_char_dataset, plot_loss
from models.language.mlp_torch import Linear, BatchNorm1d, Tanh


# https://www.youtube.com/watch?v=t3YJ5hKiMQ0
# https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb


class Embedding:
  """
  Embeds characters into a vector space.
  Each character is represented by a vector of size dim_emb.
  """
  def __init__(self, vocab_size, n_emb):
    # before: c = torch.randn((vocab_size, n_emb))
    self.weight = torch.randn((vocab_size, n_emb))
    
  def __call__(self, iz):
    self.out = self.weight[iz]
    return self.out
  
  def parameters(self):
    return [self.weight]


class Flatten:
  """
  Flattens the input into a vector.
  'Appends' each character in the context (block_size) to a single vector.
  Each character is represented by a vector of size dim_emb.
  """
  def __call__(self, x):
    # input: (num_samples, len_context, dim_emb)
    # output: (num_samples, len_context*dim_emb)
    self.out = x.view(x.shape[0], -1)
    return self.out
  
  def parameters(self):
    return []
  

class FlattenConsecutive:
  """
  Flattens consecutive characters into a single vector.
  We want to group (fuse) n=2 consecutive characters at a time (what makes it a wavenet).
  """
  def __init__(self, size_group=2):
    self.size_group = size_group
    
  def __call__(self, x):
    # input: (num_samples, len_context, dim_emb)
    # output: (num_samples, num_groups, dim_emb*size_group)
    # num_groups = len_context // size_group
    num_samples, len_context, dim_emb = x.shape
    # if len_context//self.size_group == 1, then we don't need to group characters
    x = x.view(num_samples, len_context//self.size_group, dim_emb*self.size_group)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
    return self.out
  
  def parameters(self):
    return []
  

class Sequential:
  """A sequential container for layers = model."""
  def __init__(self, layers):
    self.layers = layers
  
  def __call__(self, x):
    # forward pass
    # feed input through all layers
    for layer in self.layers:
      # print(f'layer: {layer.__class__.__name__}')
      x = layer(x)
      # print(f'  output shape: {x.shape}')
    self.out = x
    return self.out
  
  def parameters(self):
    # get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]
  

class Wavenet:
  """
  Predicts the next character based on a couple of previous characters.
  Wavenet = convolutional neural network.
  """
  def __init__(self, len_context=8, n_hidden=128, dim_emb=24, epochs=1000):
    # the dimensionality of the character embedding vectors
    self.dim_emb = dim_emb
    # the number of neurons in the hidden layer of the MLP
    self.n_hidden = n_hidden
    # the number of characters in a context
    self.len_context = len_context
    # load data
    words, self.token, self.vocab_size, self.chr_to_int, self.int_to_chr = load_names()
    x, y = build_char_dataset(words, self.len_context, self.chr_to_int)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # init model
    self.model, self.parameters = self.init_model()
    # train model
    losses_log = self.train(x_train, y_train, epochs=epochs)
    self.plot_loss(losses_log, epochs=epochs)
    # test model
    self.evaluate_model(x_train, y_train, x_test, y_test)
    self.sample_predictions()

  
  def init_model(self, size_group=2):
    vocab_size = self.vocab_size
    dim_emb = self.dim_emb
    n_hidden = self.n_hidden
    # Standard MLP
    # model = Sequential([
    #   Embedding(vocab_size, dim_emb),
    #   FlattenConsecutive(8), Linear(dim_emb * self.blocksize, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    #   Linear(n_hidden, vocab_size),
    # ])
    # Hierarchical network
    # test if len_context is a power of size_group
    test = m.log(self.len_context, size_group)
    if test != int(test):
      # we group size_group consecutive characters, out of len_context characters
      # Groups of 2 characters, 
      # then groups of 2 of groups of 2 characters, 
      # then groups of 2 of groups of 2 of groups of 2 characters, …
      raise ValueError(f'len_context must be a power of {size_group}')
    # build model
    layer_list = [
      Embedding(vocab_size, dim_emb),
      FlattenConsecutive(size_group),
      Linear(dim_emb * size_group, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh()]
    for _ in range(int(test)-1):
      layer_list += [
        FlattenConsecutive(size_group),
        Linear(n_hidden*size_group, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh()
      ]
    layer_list.append(Linear(n_hidden, vocab_size))
    model = Sequential(layer_list)
    # explicit example
    # if self.len_context == 8:
    #   model = Sequential([
    #     # (num_samples, len_context)
    #     Embedding(vocab_size, dim_emb), # (vocab_size, n_emb)[(num_samples, len_context)] 
    #     # (num_samples, len_context, dim_emb)

    #     FlattenConsecutive(size_group), 
    #     # (num_samples, len_context//size_group, dim_emb*size_group)
    #     Linear(dim_emb * size_group, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(), 
    #     # (num_samples, len_context//size_group, n_hidden)

    #     FlattenConsecutive(size_group),
    #     # (num_samples, len_context//(size_group*size_group), n_hidden*size_group)
    #     Linear(n_hidden*size_group, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    #     # (num_samples, len_context//(size_group**2), n_hidden)

    #     FlattenConsecutive(size_group), 
    #     # (num_samples, len_context//(size_group**2), n_hidden*size_group)
    #     Linear(n_hidden*size_group, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    #     # (num_samples, len_context//(size_group**3), n_hidden)

    #     Linear(n_hidden, vocab_size),
    #     # (num_samples, len_context//(size_group**3), vocab_size)
    #   ])
    # parameter init
    with torch.no_grad():
      model.layers[-1].weight *= 0.1 # last layer make less confident
    # parameters
    parameters = model.parameters()
    print('Parameters in model:', sum(p.nelement() for p in parameters)) # number of parameters in total
    for p in parameters:
      p.requires_grad = True
    return model, parameters
  
  def train(self, x_train, y_train, epochs, batch_size=32):
    # same optimization as last time
    losses_log = []
    for i in range(epochs):
      # minibatch construct
      batch = torch.randint(0, x_train.shape[0], (batch_size,))
      # (batch_size, len_context), (batch_size,)
      x_batch, y_batch = x_train[batch], y_train[batch] 
      # forward pass
      logits = self.model(x_batch)
      loss = F.cross_entropy(logits, y_batch) # loss function
      # backward pass
      for p in self.parameters:
        p.grad = None
      loss.backward()
      # update
      lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
      for p in self.parameters:
        p.data += -lr * p.grad
      # track stats
      if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{epochs:7d}: {loss.item():.4f}')
      losses_log.append(loss.log10().item())
    return losses_log

  def evaluate_model(self, x_train, y_train, x_test, y_test):
    # put layers into eval mode (needed for batchnorm especially)
    for layer in self.model.layers:
      layer.training = False
    # evaluate the loss
    @torch.no_grad() # this decorator disables gradient tracking inside pytorch
    def split_loss(split):
      x, y = {
        'train': (x_train, y_train),
        'test': (x_test, y_test),
      }[split]
      logits = self.model(x)
      loss = F.cross_entropy(logits, y)
      print(split, loss.item())
    split_loss('train')
    split_loss('test')
  
  def sample_predictions(self, num_samples=20):
    samples = []
    for _ in range(num_samples):
      out = []
      len_context = [0] * self.len_context # initialize with all ...
      while True:
        # forward pass the neural net
        logits = self.model(torch.tensor([len_context]))
        probs = F.softmax(logits, dim=1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1).item()
        # shift the len_context window and track the samples
        len_context = len_context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, break
        if ix == 0:
          break
      samples.append(''.join(self.int_to_chr[i] for i in out)) # decode and print the generated word
    print('samples:\n', *samples)
    return samples