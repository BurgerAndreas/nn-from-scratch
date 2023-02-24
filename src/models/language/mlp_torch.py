import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from helpers.get_data import load_names, train_test_split, build_char_dataset

#  https://youtu.be/P6sfmUTpUmc?t=4713
# https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb


class Linear:
  """Linear Layer."""
  def __init__(self, fan_in, fan_out, bias=True):
    """Initialize the weights and bias."""
    # fan_in: number of input neurons
    # fan_out: number of output neurons
    # Kaiman Initialization
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
    # bias initialized to 0
    self.bias = torch.zeros(fan_out) if bias else None 
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
  """Batch Normalization."""
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps # epsilon for numerical stability
    self.momentum = momentum # momentum for the buffers
    self.training = True
    # parameters (trained with backprop)
    # whats returned by the layer
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    # moving averages of the mean and variance
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    """Get the output of the layer (gamma * xhat + beta).
    When training, update the buffers."""
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      # only update buffers when training
      with torch.no_grad():
        # we don't want to track the gradients for the buffers
        # buffers are not trained with backprop
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]


class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []
  

class CharMLP2:
  """
  Predicts the next character based on a couple of previous characters.
  consists of classes very similar to the ones in the PyTorch API.
  Includes Kaiman Initialization, Batch Normalization.
  """
  # def __init__(self, vocab_size, hidden_size, num_layers=1, bias=True):
  #   self.vocab_size = vocab_size
  #   self.hidden_size = hidden_size
  #   self.num_layers = num_layers
  #   self.bias = bias
  #   # create the layers
  #   self.layers = []
  #   for i in range(num_layers):
  #     if i == 0:
  #       self.layers.append(Linear(vocab_size, hidden_size, bias))
  #     else:
  #       self.layers.append(Linear(hidden_size, hidden_size, bias))
  #     self.layers.append(BatchNorm1d(hidden_size))
  #     self.layers.append(Tanh())
  #   self.layers.append(Linear(hidden_size, vocab_size, bias))
  #   self.layers.append(BatchNorm1d(vocab_size))
  
  # def __call__(self, x):
  #   for layer in self.layers:
  #     x = layer(x)
  #   return x
  
  # def parameters(self):
  #   params = []
  #   for layer in self.layers:
  #     params += layer.parameters()
  #   return params
  

  def __init__(self, block_size=3, n_hidden=100, n_embd=10, batch_norm=True):
    # the dimensionality of the character embedding vectors
    self.n_embd = n_embd
    # the number of neurons in the hidden layer of the MLP
    self.n_hidden = n_hidden
    # the number of characters in a context
    self.block_size = block_size
    # whether to use batch normalization
    self.batch_norm = batch_norm
    # load data
    words, self.token, self.num_tokens, self.chr_to_int, self.int_to_chr = load_names()
    # the number of characters in the vocabulary = alphabet + start/stop token
    vocab_size = self.num_tokens
    # initialize the model parameters
    self.c, self.layers, self.parameters = self.init_model(vocab_size, block_size, n_hidden, n_embd, batch_norm)
    # do stuff
    self.train_and_test(words)
  
  def train_and_test(self, words, epochs=1000):
    x, y = build_char_dataset(words, self.block_size, self.chr_to_int)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    lossi, ud = self.train(x_train, y_train, epochs=epochs)
    # visualize
    self.visualize_loss(lossi)
    self.visualize_step_size(ud)
    self.visualize_activations()
    self.visualize_gradients()
    self.visualize_weigth_gradients()
    # evaluate
    self.evaluate_model(x_train, y_train, x_test, y_test)
    self.sample_predictions()


  def init_model(self, vocab_size, block_size, n_hidden, n_embd, batch_norm):
    """Initialize the model parameters.
    Character embedding and layers."""
    c = torch.randn((vocab_size, n_embd))
    if batch_norm:
      # BatchNorm could also be after non-linearity
      layers = [
        Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
      ]
    else:
      layers = [
        Linear(n_embd * block_size, n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden), Tanh(),
        Linear(           n_hidden, vocab_size),
      ]
    # initialize the weights
    with torch.no_grad():
      # last layer: make less confident
      layers[-1].gamma *= 0.1
      #layers[-1].weight *= 0.1
      # all other layers: apply gain
      for layer in layers[:-1]:
        if isinstance(layer, Linear):
          # gain to stabilize the variance of the activations
          # roughly 0.6 for tanh (see visualize_activations())
          # tanh squashes, this counters the squashing
          # kaiming initialization
          layer.weight *= 5/3 
    # all parameters
    parameters = [c] + [p for layer in layers for p in layer.parameters()]
    print(sum(p.nelement() for p in parameters)) # number of parameters in total
    for p in parameters:
      p.requires_grad = True
    return c, layers, parameters

  def train(self, x_train, y_train, epochs, batch_size=32):
    # same optimization as last time
    lossi = []
    ud = [] # update size to data size ratio = relative step size
    for i in range(epochs):
      # minibatch construct
      ix = torch.randint(0, x_train.shape[0], (batch_size,))
      Xb, Yb = x_train[ix], y_train[ix] # batch X,Y
      # forward pass
      emb = self.c[Xb] # embed the characters into vectors
      x = emb.view(emb.shape[0], -1) # concatenate the vectors
      for layer in self.layers:
        x = layer(x)
      loss = F.cross_entropy(x, Yb) # loss function
      # backward pass
      for layer in self.layers:
        layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
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
      lossi.append(loss.log10().item())
      with torch.no_grad(): # relative step size
        ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in self.parameters])
    return lossi, ud
  

  def evaluate_model(self, x_train, y_train, x_test, y_test):
    @torch.no_grad() # this decorator disables gradient tracking
    def split_loss(split):
      x, y = {
        'train': (x_train, y_train),
        'test': (x_test, y_test),
      }[split]
      emb = self.c[x] # (N, block_size, n_embd)
      x = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
      for layer in self.layers:
        x = layer(x)
      loss = F.cross_entropy(x, y)
      print(split, loss.item())
    # put layers into eval mode
    for layer in self.layers:
      layer.training = False
    split_loss('train')
    split_loss('test')
    return 

  def sample_predictions(self, num_samples=20):
    samples = []
    for _ in range(num_samples):
      out = []
      context = [0] * self.block_size # initialize with all ...
      while True:
        # forward pass the neural net
        emb = self.c[torch.tensor([context])] # (1,block_size,n_embd)
        x = emb.view(emb.shape[0], -1) # concatenate the vectors
        for layer in self.layers:
          x = layer(x)
        logits = x
        probs = F.softmax(logits, dim=1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1).item()
        # shift the context window and track the samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token, break
        if ix == 0:
          break
      samples.append(''.join(self.int_to_chr[i] for i in out)) # decode and print the generated word
    print('samples:\n', *samples)
    return samples
    
  def visualize_loss(self, losses):
    plt.plot(losses)
    plt.title('loss')
    plt.show()

  def visualize_activations(self):
    # histograms
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i, layer in enumerate(self.layers[:-1]): # note: exclude the output layer
      if isinstance(layer, Tanh):
        t = layer.out
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__}')
    plt.legend(legends)
    plt.title('activation distribution for tanh layers (forward pass)')
    plt.show()

  def visualize_gradients(self):
    # visualize histograms
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i, layer in enumerate(self.layers[:-1]): # note: exclude the output layer
      if isinstance(layer, Tanh):
        t = layer.out.grad
        print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__}')
    plt.legend(legends);
    plt.title('gradient distribution for tanh layers (backward pass)')
    plt.show()
  
  def visualize_weigth_gradients(self):
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i, p in enumerate(self.parameters):
      if p.ndim == 2:
        t = p.grad
        print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'{i} {tuple(p.shape)}')
    plt.legend(legends)
    plt.title('weights gradient distribution')
    plt.show()

  def visualize_step_size(self, ud):
    # stabalizes over time
    plt.figure(figsize=(20, 4))
    legends = []
    for i,p in enumerate(self.parameters):
      if p.ndim == 2:
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append('param %d' % i)
    # these ratios should be ~1e-3, indicated on plot
    # if lr is too low, the ratio will be below 1e-3
    plt.plot([0, len(ud)], [-3, -3], 'k') 
    plt.legend(legends)
    plt.show()
    