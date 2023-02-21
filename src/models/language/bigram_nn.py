import torch
import matplotlib.pyplot as plt

from helpers.get_data import load_names, g

# https://youtu.be/PaCmpygFfXo


class BigramSLNN:
  """Character bigram model. 
  Predicts the next character in a word based on only the previous character.
  Single layer neural network.
  Should converge to the same result as the bigram counter."""

  def __init__(self):
    """Initializes the model."""
    self.words, self.token, self.num_tokens, self.chr_to_int, self.int_to_chr = load_names()
    x_train, y_train = self.get_train_data()
    x_train_enc = self.input_encoding(x_train)
    self.w = self.layer()
    self.train(x_train_enc, y_train)


  def get_train_data(self):
    # training set, first and second character of bigram
    x_train, y_train = [], []
    # itarate over all words and b igrams
    for w in self.words:
      # start and end token
      chs = [self.token] + list(w) + [self.token]
      for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = self.chr_to_int[ch1]
        ix2 = self.chr_to_int[ch2]
        x_train.append(ix1)
        y_train.append(ix2)
    # convert to tensors
    # int64 is the default for torch.tensor and needed for one_hot
    x_train = torch.tensor(x_train, dtype=torch.int64) 
    y_train = torch.tensor(y_train)
    return x_train, y_train


  def input_encoding(self, x_train):
    """Input encoding"""
    # one-hot input encoding
    # (num_samples, num_tokens) = [228146, 27]
    return torch.nn.functional.one_hot(x_train, num_classes=self.num_tokens).float()
  

  def layer(self):
    """A single layer. 
    Layer is input and output layer."""
    # 27 neurons with 27 inputs each
    # could also be just one neuron with 27 inputs
    num_neurons = self.num_tokens  
    # initialize weights at random
    # (num_tokens, num_neurons) = [27, 27]
    return torch.randn((self.num_tokens, num_neurons), requires_grad=True)


  def train(self, x_train_enc, y_train, epochs=100, learning_rate=50):
    num_train_samples = x_train_enc.shape[0]
    loss = 100
    for i in range(epochs):
      # forward pass
      # (num_samples, num_neurons) = [228146, 27]
      logits = x_train_enc @ self.w # log-counts
      # softmax = exponentiate and normalize
      counts = torch.exp(logits) # same as counts in simple bigram model
      probs = counts / counts.sum(dim=1, keepdim=True)
      # loss
      # L2 regularization helps to prevent overfitting 
      # and with 0 counts (unseen bigrams in training)
      # like model smoothing in the counting bigram model
      regularization = 0.01 * (self.w**2).mean()
      # range(num_train_samples) picks out probabilities for next character
      loss = -torch.log(probs[range(num_train_samples), y_train]).mean() + regularization
      # backpropagation
      loss.backward()
      # update weights
      with torch.no_grad():
        self.w -= learning_rate * self.w.grad
        self.w.grad = None
      # print loss
      if i % 1 == 0:
        print(f'epoch: {i}, loss: {loss:.3f}')
    return loss
  

  def predict(self, num_samples=20):
    """Sample from NN. By sampling the next character."""
    samples = []
    for _ in range(num_samples):
      sample = []
      # start token
      ch = self.token
      while (ch != self.token) or not sample:
        # encode previous character
        ch_enc = self.input_encoding(torch.tensor([self.chr_to_int[ch]]))
        # forward pass
        logits = ch_enc @ self.w
        counts = torch.exp(logits)
        probs = counts / counts.sum()
        # sample next character from probabilities
        ch = torch.multinomial(probs, num_samples=1, generator=g).item()
        # convert to character
        ch = self.int_to_chr[ch]
        # add to sample
        sample.append(ch)
      samples.append(''.join(sample))
    return samples


def test_nn_bigram_model():
  """Tests character bigram model."""
  model = BigramSLNN()
  print(model.predict())
  return model