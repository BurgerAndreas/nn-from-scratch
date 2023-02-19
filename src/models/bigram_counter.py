import torch
import matplotlib.pyplot as plt

# https://youtu.be/PaCmpygFfXo

# generator for randomness for reproducability
g = torch.Generator().manual_seed(0)

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


class BigramCounter:
  """Character bigram model. 
  Predicts the next character in a word based on only the previous character.
  Model works by counting occurances of bigrams.
  Not a neural network."""

  def __init__(self):
    """Initializes the model."""
    self.words, self.token, self.num_tokens, self.chr_to_int, self.int_to_chr = load_names()
    self.bgm_cnt = self.count_bigrams()
    self.prob_bgm = self.probabilities_bigrams()


  # What does the dataset look like?
  def print_dataset_info(self):
    """Prints dataset information."""
    n_words = len(self.words)
    min_len = min(len(w) for w in self.words)
    max_len = max(len(w) for w in self.words)
    print(f'Number of self.words: {n_words}')
    print(f'Minimum word length: {min_len}')
    print(f'Maximum word length: {max_len}')
    return


  def count_bigrams(self):
    # count of bigrams 
    bgm_cnt = torch.zeros((self.num_tokens, self.num_tokens), dtype=torch.int32)
    # itarate over all bigrams
    for w in self.words:
      # start and end token
      chs = [self.token] + list(w) + [self.token]
      for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = self.chr_to_int[ch1]
        ix2 = self.chr_to_int[ch2]
        # Increment count of bigram
        bgm_cnt[ix1, ix2] += 1
    return bgm_cnt


  def plot_bigram_counts(self):
    """Plots bigram counts."""
    # plt.imshow(bgm_cnt)
    plt.figure(figsize=(16, 16))
    plt.imshow(self.bgm_cnt, cmap='Blues')
    for i in range(self.num_tokens):
      for j in range(self.num_tokens):
        chstr = self.int_to_chr[i] + self.int_to_chr[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, self.bgm_cnt[i,j].item(), ha='center', va='top', color='gray')
    plt.axis('off')
    plt.show()
    return


  def probabilities_bigrams(self):
    """Converts bigram counts to probabilities."""
    # Model smoothing (increase all counts by 1, to avoid zero probabilities)
    bgm_cnt = self.bgm_cnt + 1
    # convert bigram counts to probabilities
    prob_bgm = bgm_cnt.float()
    # sum over rows, but keep dimensions
    # s.t. broadcasting works
    prob_bgm = prob_bgm / prob_bgm.sum(dim=1, keepdim=True)
    return prob_bgm
  

  def sample_new_words(self, num_samples=10):
    """Prints bigram samples."""
    # generate samples
    samples = []
    for _ in range(num_samples):
      sample = []
      ix = 0 # start token
      while True:
        # convert bigram counts to probabilities
        prob_char = self.prob_bgm[ix]
        # prob_char = bgm_cnt[ix].float()
        # prob_char = prob_char / prob_char.sum()
        # compare to uniform distribution
        # prob_char = torch.ones(num_tokens) / num_tokens
        # sample next character
        ix = torch.multinomial(prob_char, num_samples=1, replacement=True, generator=g).item()
        sample.append(self.int_to_chr[ix])
        if ix == 0: # end token
          break
      print(''.join(sample))
      samples.append(''.join(sample))
    return samples


  def evaluate_model(self, num_samples=-1, test_sample=None):
    """Evaluates model."""
    # random_prob = 1 / num_tokens
    # print(f'Random probability: {random_prob:.3f} {torch.log(torch.tensor(random_prob)).item():.3f}')
    log_likelyhood = 0
    n_bigram_samples = 0
    if not test_sample:
      test_sample = self.words
    for w in test_sample[:num_samples]:
      # start and end token
      chs = [self.token] + list(w) + [self.token]
      for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = self.chr_to_int[ch1]
        ix2 = self.chr_to_int[ch2]
        # probability of bigram
        prob = self.prob_bgm[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelyhood += log_prob
        n_bigram_samples += 1
        if num_samples < 20:
          print(f'{ch1} -> {ch2}: {prob.item():.3f} {log_prob.item():.3f}')
    # negative log likelyhood normalized
    nll = -log_likelyhood / n_bigram_samples
    print(f'Average Negative Log likelyhood: {nll.item():.3f}')
    return nll


def test_counting_bigram_model():
  """Tests counting-based character bigram model."""
  model = BigramCounter()
  model.print_dataset_info()
  model.plot_bigram_counts()
  model.sample_new_words()
  # Overall model performance
  model.evaluate_model()
  # Test model performance on test set
  model.evaluate_model(test_sample=['Hans', 'Peter', 'Paul', 'Hans-Peter', 'Andreas'])
  # Test model performance on first few words
  model.evaluate_model(num_samples=10)
  return model