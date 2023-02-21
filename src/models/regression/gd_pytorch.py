import torch

from helpers.get_data import create_linear_data, train_test_split

def torch_gd_mse_nn(learning_rate = 0.1, momentum = 0.9, epochs = 10):
  """Train a linear regression model.
  With tensorflow gradient descent optimizer.
  Use Mean-Squared Error loss."""
  # parameters
  batch, dim_in, dim_h, dim_out = 64, 1000, 100, 10
  # create random data
  input_X = torch.randn(batch, dim_in)
  output_Y = torch.randn(batch, dim_out)
  # x, y_true, w_true, b_true = create_linear_data(dim=1, num_samples=1000)
  # Model
  sgd_model = torch.nn.Sequential(
    torch.nn.Linear(dim_in, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
  )
  # Loss function
  loss_fn = torch.nn.MSELoss(reduction='sum')
  # Optimizer
  opt = torch.optim.SGD(sgd_model.parameters(), lr=learning_rate, momentum=momentum)
  #  Forward pass
  for epoch in range(epochs):
    y_pred = sgd_model(input_X)
    loss = loss_fn(y_pred, output_Y)
    # Backward pass
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f'Epoch {epoch}:', loss.item())
  return sgd_model


def torch_gd_mse_lr(learning_rate = 0.1, momentum = 0.9, epochs = 100):
  """Train a linear regression model."""
  # create data
  X = torch.arange(-5, 5, 0.1).view(-1, 1)
  func = -5 * X
  Y = func + 0.4 * torch.randn(X.size())
  
  # defining the function for forward pass for prediction
  def forward(x):
    return w * x + b
  
  # evaluating data points with Mean Square Error (MSE)
  def mse_loss(y_pred, y):
      return torch.mean((y_pred - y) ** 2)
  
  w = torch.tensor(-10.0, requires_grad=True)
  b = torch.tensor(-20.0, requires_grad=True)
  
  loss_BGD = []
  
  for i in range (epochs):
    # making predictions with forward pass
    Y_pred = forward(X)
    # calculating the loss between original and predicted data points
    loss = mse_loss(Y_pred, Y)
    # storing the calculated loss in a list
    loss_BGD.append(loss.item())
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updateing the parameters after each iteration
    w.data -= - learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data
    # zeroing gradients after each iteration
    w.grad.data.zero_()
    b.grad.data.zero_()
    print(f'Epoch {i}:', loss.item())


# def torch_gd_mse_qr(learning_rate=0.1, momentum=0.9, epochs=10):
#   """Train a quadratic regression model."""
#   # model
#   def f(x, params):
#      a,b,c = params
#      return a*(x**2) + (b*x) + c
#   # loss function
#   def mse(preds, targets): 
#     return ((preds-targets)**2).mean().sqrt()
#   # initialize weights
#   params = torch.randn(3).requires_grad_()
#   preds = f(x, params)
#   # show initial 
#   loss = mse(preds, speed)
#   loss.backward()
#   # train model
#   for epoch in range(epochs):
#     # params.grad
#     params.data -= learning_rate * params.grad.data
#     params.grad = None


