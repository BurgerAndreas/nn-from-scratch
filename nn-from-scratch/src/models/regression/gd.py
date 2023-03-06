import numpy as np

from src.helpers.get_data import create_linear_data, train_test_split

class GradientDescentMSELinearRegression:
  """Train a linear regression model.
  Eith gradient descent on a Mean-Squared Error loss.
  Build from scratch."""

  def __init__(self):
    # create data
    x, y_true, w_true, b_true = create_linear_data(dim=1, num_samples=1000)
    X_train, X_test, y_train, y_test = self.train_test_split(x, y_true, test_size=0.2)
    # train model
    w_pred = self.gradient_descent_training(X_train, y_train, batch_size=100, learning_rate=0.8, momentum=0.1, epochs=100)
    # test model
    self.validation_test(X_test, y_test, w_pred)

  def partial_derivative(self, X_batch, y_batch, w_pred):
    """Calculate partial derivative (for a batch of samples)."""
    # f(x) = y_pred = w_pred * x + b_pred
    # b_pred is absorbed into w_pred
    y_pred = X_batch @ w_pred
    n_samples = len(X_batch) 
    # mse = 1/n * sum((y - y_pred)**2)
    # d(mse)/d(w_pred) = -2/n * sum(x * (y - y_pred(w_pred)))
    df_dw =  (-2/n_samples) * (X_batch.T @ (y_batch - y_pred))
    df_dw = df_dw.reshape(len(df_dw), -1)
    return df_dw
    
  def calc_loss(self, x, y_true, w_pred):
    """Calculate loss."""
    y_pred = x @ w_pred
    # # negative log likelihood
    # loss = -np.mean(y_true * np.log(y_pred))
    # mean squared error
    # lower is better
    loss = np.mean((y_true - y_pred)**2)
    loss = np.sum(((y_true - y_pred)**2), axis=0) / len(y_true)
    return loss

  # stochastic gradient descent
  def gradient_descent_training(self, x_train, y_train, batch_size=10, learning_rate=0.1, momentum=0., epochs=100, stop_threshold=1e-06):
    # initialize weights at random
    low, high = -1, 1
    w_pred = np.random.rand(x_train.shape[1], 1) * (high - low) + low
    for epoch in range(epochs):
      # pick random samples for batches (stochastic)
      # via shuffle of X and y (permutation)
      indices = np.arange(x_train.shape[0])
      np.random.shuffle(indices)
      x_train = x_train[indices]
      y_train = y_train[indices]
      # differences in weights between consecutive batches
      update = 0
      for batch in range(len(x_train) // batch_size):
        start = batch*batch_size
        X_batch = x_train[start:start + batch_size]
        y_batch = y_train[start:start + batch_size]
        # 
        update = (momentum*update) - (learning_rate*self.partial_derivative(X_batch, y_batch, w_pred))
        # if updates are too small, stop training
        if np.all(np.abs(update) < stop_threshold):
          break
        # update weights after each batch
        # w_pred -= learning_rate * partial_derivative(X_batch, y_batch, w_pred)
        w_pred += update
      print(f"epoch: {epoch} ----> MSE: {self.calc_loss(x_train, y_train, w_pred)}")  
    return w_pred

  def validation_test(self, x_test, y_test, w_pred):
    loss = self.calc_loss(x_test, y_test, w_pred)
    print(f"Test set (validation) loss ----> MSE: {loss}")  
    return loss



