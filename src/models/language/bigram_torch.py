import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from helpers.get_data import load_text, train_val_split_text, plot_loss
from models.language.transformer import estimate_loss, get_batch


# https://youtu.be/kCc8FmEb1nY?t=2459
# https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py


class ShakespeareBigram(nn.Module):
  """Super simple bigram model for Shakespeare text generation.
  Benchmark for transformer model."""
  def __init__(self):
    super().__init__()
    # hyperparameters
    # 4.225 K parameters
    # step 0: train loss 4.5783, val loss 4.5743  
    # step 1200: train loss 2.4842, val loss 2.5083
    # training took 7.18 seconds
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 8 # what is the maximum context length for predictions?
    epochs = 3000
    learning_rate = 1e-2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_batches = 200
    # ------------
    # data loading
    text, alphabet, vocab_size, encode, decode = load_text()
    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, val_data = train_val_split_text(data, train_size=0.9)
    # ------------
    # model
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) 
    # ------------
    self.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in self.parameters())/1e3, 'K parameters')
    # create a PyTorch optimizer
    self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
    # ------------
    # training loop
    losses = self.train_model(train_data, val_data, block_size, batch_size, device, epochs, eval_batches)
    plot_loss(losses, n_points=-1)
    # sample from the model
    self.sample(decode, block_size, device, max_new_tokens=200)

  def forward(self, context, targets=None):
    # context and targets are both (B,T) tensor of integers
    logits = self.token_embedding_table(context) # (B,T,C)
    if targets is None:
      loss = None
    else:
      # Batch, Time, Channel
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def train_model(self, train_data, val_data, block_size, batch_size, device, epochs, eval_batches):
    t_start = time.time()
    losses_iter = []
    for iter in range(epochs):
      # every once in a while evaluate the loss on train and val sets
      if iter % (epochs/10) == 0 or iter == epochs - 1:
        losses = estimate_loss(self, train_data, val_data, block_size, batch_size, device, eval_batches)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
      # sample a batch of data
      xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device)
      # evaluate the loss
      logits, loss = self(xb, yb)
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      self.optimizer.step()
      losses_iter.append(loss.item())
    print(f"training took {time.time() - t_start:.2f} seconds")
    return losses_iter
  
  def sample(self, decode, block_size, device, context=None, max_new_tokens=200):
    if not context:
      context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # context is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # get the predictions
      logits, loss = self(context)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from the distribution
      context_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
      context = torch.cat((context, context_next), dim=1) # (B, T+1)
    # decode the context
    # context[0] remove empty dimension
    text = decode(context[0].tolist())
    print(text)
    return text