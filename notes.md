# What every ML model needs

- Data
- Model
- Loss
- Optimizer

### Data

With or without labels

### Model

Linear Regression, MLP, Transformer

### Loss

MSE, NLL, (Binary) Cross-Entropy, Regularization

### Optimizer

Stochastic/Momentum Gradient Descent, Momentum, Nesterov, Adam 


# Initializations

see kaiming initialization

```
# gaussian with std 1
w = torch.randn((in, out), requires_grad=True)) 

# gaussian with std 0.2
w = torch.randn((in, out), requires_grad=True)) * 0.2  

# = roughly kaiming initialization for tanh
nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
```

# Activations, Gradients

Bad distribution at output layer: hockey stick loss.
If activations are too extreme (squeezed to 0 or at infinities), gradients will be too small (vanishing gradient problem).
You want roughly gaussian distribution over neural net.

### Batch Norm Layer

Batch Norm stablisizes training and improves generalization.

Normalize hidden states to be unit gaussian (mean 0 and std 1). 
Done for each batch separately. 
Since batches are sampled stochastically, it will add noise, and act as a form of regularization.

Since moving to unit gaussian depends on the other examples in the batch, it will couple the examples in the batch together mathematically.
This is problematic.

Mean and std are often calculated with exponential moving average ('running') over the training set.
This is reversible, and can be used at backprop.
