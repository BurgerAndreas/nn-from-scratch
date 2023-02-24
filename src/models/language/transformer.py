import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from helpers.get_data import load_text, train_val_split_text, plot_loss

# https://www.youtube.com/watch?v=kCc8FmEb1nY

# ------------
# also used in bigram_torch.py
# better as a class method, and bigram should inherit from this class?
def get_batch(split, train_data, val_data, block_size, batch_size, device):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[ i : i+block_size ] for i in ix])
  y = torch.stack([data[ i+1 : i+block_size+1 ] for i in ix])
  x, y = x.to(device), y.to(device)
  # x is (batch_size, block_size) and y is (batch_size, block_size)
  # batch_size is the number of independent sequences we are processing in parallel
  # block_size is the maximum context length for predictions = examples+1 = time dimension
  return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_batches=100):
  """Estimeate loss just over a couple of batches"""
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_batches)
    for k in range(eval_batches):
      x, y = get_batch(split, train_data, val_data, block_size, batch_size, device)
      # (batch_size, block_size)
      logits, loss = model(x, y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  # sets training mode. e.g. Dropout and BatchNorm
  model.train()
  return out


# -------------------------------------------------------------------------------
def get_hyperparameters(model_size='small'):
  if model_size == 'small':
    # 29.761 K parameters
    # step 0: train loss 4.1849, val loss 4.1863
    # step 999: train loss 2.7555, val loss 2.7509
    # training took 14.58 seconds
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 8 # what is the maximum context length for predictions?
    epochs = 1000
    learning_rate = 3e-2
    eval_batches = 200
    n_emb = 32
    n_head = 2
    n_layer = 2
  elif model_size == 'medium':
    # 45s training time
    # 208.193 K parameters
    # step 0: train loss 4.2318, val loss 4.2326
    # step 999: train loss 2.3284, val loss 2.3485
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 8 # what is the maximum context length for predictions?
    epochs = 1000
    learning_rate = 3e-3
    eval_batches = 200
    n_emb = 64
    n_head = 4
    n_layer = 4
  elif model_size == 'medium_l_emb':
    # larger embedding doesn't improve on medium
    # 3191.873 K parameters
    # step 0: train loss 4.1964, val loss 4.1936
    # step 100: train loss 2.6912, val loss 2.7068
    # step 999: train loss 2.5249, val loss 2.5372
    # training took 215.41 seconds
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 8 # what is the maximum context length for predictions?
    epochs = 1000
    learning_rate = 3e-3
    eval_batches = 200
    n_emb = 64 * 4 # vocab_size * n_head
    n_head = 4
    n_layer = 4
  elif model_size == 'medium_l_context':
    # larger context improves medium significantly
    # 211.777 K parameters
    # step 0: train loss 4.1766, val loss 4.1784
    # step 999: train loss 1.9102, val loss 2.0034
    # training took 232.31 seconds
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 64 # what is the maximum context length for predictions?
    epochs = 1000
    learning_rate = 3e-3
    eval_batches = 200
    n_emb = 64
    n_head = 4
    n_layer = 4
  elif model_size == 'large':
    # 205.633 K parameters
    # step 0: train loss 4.1938, val loss 4.1957
    # step 999: train loss 2.0030, val loss 2.0671
    # training took 100.30 seconds
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 32 # what is the maximum context length for predictions?
    epochs = 1000
    learning_rate = 3e-3
    eval_batches = 200
    n_emb = 64
    n_head = 6
    n_layer = 4
  elif model_size == 'larger_emb':
    # again larger embedding doesn't improve on large
    # 808.513 K parameters
    # step 0: train loss 4.2250, val loss 4.2234
    # step 999: train loss 2.0943, val loss 2.1461
    # training took 173.30 seconds
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 32 # what is the maximum context length for predictions?
    epochs = 1000
    learning_rate = 3e-3
    eval_batches = 200
    n_emb = 128
    n_head = 6
    n_layer = 4
  elif model_size == 'larger_context':
    # again larger context improves large
    # 207.681 K parameters
    # step 0: train loss 4.1960, val loss 4.1959
    # step 999: train loss 1.9135, val loss 2.0151
    # training took 141.90 seconds
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 64 # what is the maximum context length for predictions?
    epochs = 1000
    learning_rate = 3e-3
    eval_batches = 200
    n_emb = 64
    n_head = 6
    n_layer = 4
  elif model_size == 'big':
    # even larger context gives better predictions
    # 211.777 K parameters
    # step 0: train loss 4.1702, val loss 4.1683
    # step 999: train loss 1.8246, val loss 1.9582
    # training took 589.14 seconds
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 128 # what is the maximum context length for predictions?
    epochs = 1000
    learning_rate = 3e-3
    eval_batches = 200
    n_emb = 64
    n_head = 6 
    n_layer = 4
  elif model_size == 'big_lessheads':
    # more parameters because each head is bigger?
    # 215.744 K parameters
    # step 0: train loss 4.1949, val loss 4.1954
    # step 999: train loss 1.8114, val loss 1.9435
    # training took 478.60 seconds
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 128 # what is the maximum context length for predictions?
    epochs = 1000
    learning_rate = 3e-3
    eval_batches = 200
    n_emb = 64
    n_head = 4 
    n_layer = 4
  elif model_size == 'huge':
    # 309.313 K parameters
    # step 0: train loss 4.1749, val loss 4.1749
    # step 999: train loss 1.8368, val loss 1.9848
    # training took 478.08 seconds
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 128 # what is the maximum context length for predictions?
    epochs = 1000
    learning_rate = 3e-3
    eval_batches = 200
    n_emb = 64
    n_head = 6
    n_layer = 6
  elif model_size == 'karpathy':
    # inf training time
    batch_size = 64 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    epochs = 5000
    learning_rate = 3e-4
    eval_batches = 200
    n_emb = 384  # = vocab_size * n_head
    n_head = 6
    n_layer = 6
  else:
    raise ValueError(model_size, 'invalid')
  return batch_size, block_size, epochs, learning_rate, eval_batches, n_emb, n_head, n_layer
  

# -------------------------------------------------------------------------------
class Head(nn.Module):
  """ one head of self-attention (decoder block)"""
  def __init__(self, head_size, n_emb, block_size, dropout):
    super().__init__()
    # each token has a query, a key, and a value
    # query = what am I looking for (in past tokens)
    self.query = nn.Linear(n_emb, head_size, bias=False)
    # key = what do I represent (to future tokens)
    self.key = nn.Linear(n_emb, head_size, bias=False)
    # TODO
    # value = value of token for this head
    # v instead of x. x is 'private'
    self.value = nn.Linear(n_emb, head_size, bias=False)
    # lower triangular matrix = attention towards past tokens only
    # makes it a decoder block
    # buffer = not a parameter (torch lingo)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    B,T,C = x.shape
    # hs = TODO
    k = self.key(x)   # (B,T,hs)
    q = self.query(x) # (B,T,hs)
    # compute attention scores ("affinities") = query * key
    wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    # we use value instead of x
    v = self.value(x) # (B,T,hs)
    out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
    return out


class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """
  def __init__(self, num_heads, head_size, n_emb, block_size, dropout):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, n_emb, block_size, dropout) for _ in range(num_heads)])
    # for residual connections
    # projection = linear transformation 
    # projection into residual pathway
    self.proj = nn.Linear(head_size * num_heads, n_emb)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # run multiple heads in parellel
    # concat the head output together along channel dim (still independent)
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out


class FeedFoward(nn.Module):
  """ a simple linear layer followed by a non-linearity """
  def __init__(self, n_emb, dropout):
    super().__init__()
    # from the attention paper, could also be 1
    mul = 4
    self.net = nn.Sequential(
        nn.Linear(n_emb, mul * n_emb),
        nn.ReLU(),
        # residual connections
        nn.Linear(mul * n_emb, n_emb), # delete for no residual connections
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  """ Transformer block: communication followed by computation """
  def __init__(self, n_emb, n_head, block_size, dropout):
    # n_emb: embedding dimension of tokens = channel dim
    # n_head: the number of heads
    super().__init__()
    # Attention = communication
    head_size = n_emb // n_head
    self.sa = MultiHeadAttention(n_head, head_size, n_emb, block_size, dropout)
    # MLP = computation
    self.ffwd = FeedFoward(n_emb, dropout)
    self.ln1 = nn.LayerNorm(n_emb)
    self.ln2 = nn.LayerNorm(n_emb)

  def forward(self, x):
    # residual connections
    # non residual: x = self.  
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
    

# -------------------------------------------------------------------------------
# - tokens
# characters in the alphabet + special characters = 64
# - B
# multiple batches in a tensor to compute in parallel for efficiency
# each token directly reads off the logits for the next token from a lookup table
# - T
# in a context of length block_size, there are block_size-1 examples
# context becomes bigger and bigger with examples, so we can start generating from a single token
# time = context length up to block_size-1
# - C
# vocab_size = number of possible tokens = 64
# n_emb = embedding size
# TODO should be vocab_size*n_head?
# channels = embedding size
# ------------
# nn.Module 
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# Base class for all neural network modules. Your models should also subclass this class.
# __call__()
class ShakespeareLM(nn.Module):
  """
  A generative language model.
  """
  def __init__(self, model_size='small') -> None:
    super().__init__()
    # hyperparameters
    batch_size, block_size, epochs, learning_rate, eval_batches, n_emb, n_head, n_layer = get_hyperparameters(model_size)
    dropout = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(model_size, 'model')
    # ------------
    # data loading
    text, vocabulary, vocab_size, encode, decode = load_text()
    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, val_data = train_val_split_text(data, train_size=0.9)
    print('tokens in text:', vocab_size, '\n', vocabulary)
    # ------------
    # model
    self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
    self.position_embedding_table = nn.Embedding(block_size, n_emb)
    self.blocks = nn.Sequential(*[Block(n_emb, n_head, block_size, dropout) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_emb) # final layer norm
    self.lm_head = nn.Linear(n_emb, vocab_size) # final linear layer
    # better initialization
    self.apply(self._init_weights)
    # ------------
    self.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in self.parameters())/1e3, 'K parameters')
    # create a PyTorch optimizer
    self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
    # training loop
    losses = self.train_model(train_data, val_data, block_size, batch_size, device, epochs, eval_batches)
    plot_loss(losses, n_points=-1)
    # sample from the model
    self.sample(decode, block_size, device, max_new_tokens=200)
    #
    # save and load model easily by saving their weights and biases (called state)
    torch.save(self.state_dict(), f'/models/language/saved_models/{model_size}.mdl')
    # self.load_state_dict(torch.load('ckpt.mdl'))
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  # TODO
  # called by nn.Module methods ...
  # i.e. self()
  def forward(self, context, targets=None, device='cpu'):
    # Batch, Time, Channel
    # Batch size, context length
    B, T = context.shape
    # context and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(context) # (B,T,C)
    # positional embedding, get's broadcasted over batches
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
    x = tok_emb + pos_emb # (B,T,C)
    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T,vocab_size)
    if targets is None:
      loss = None
    else:
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
      # logits, loss = model(xb, yb)
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
      # crop context to the last block_size tokens
      context_cond = context[:, -block_size:]
      # get the predictions
      logits, loss = self(context_cond)
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
