import tensorflow as tf
import numpy as np

from src.helpers.get_data import create_linear_data, train_test_split

def tf_gd_mse_lr():
  """Train a linear regression model.
  With tensorflow gradient descent optimizer.
  Use Mean-Squared Error loss."""
  # create data
  x, y_true, w_true, b_true = create_linear_data(dim=1, num_samples=1000)
  X_train, X_test, y_train, y_test = train_test_split(x, y_true, test_size=0.2)
  # create model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(1, input_shape=(2,)))
  # compile model
  model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
  # train model
  model.fit(X_train, y_train, batch_size=100, epochs=100)
  # test model
  model.evaluate(X_test, y_test)
  return model


def tf_gd(epochs=10, learning_rate=0.1, momentum=0.9):
  """ """
  # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/SGD
  # create data
  x, y_true, w_true, b_true = create_linear_data(dim=1, num_samples=1000)
  X_train, X_test, y_train, y_test = train_test_split(x, y_true, test_size=0.2)
  # create model
  # initialize weights
  w_pred = tf.Variable(np.random.randn(), name = "W")
  b_pred = tf.Variable(np.random.randn(), name = "b")
  y_pred = tf.add(tf.multiply(X_train, w_pred), b_pred)
  # First step is `- learning_rate * grad`
  # On later steps, step-size increases because of momentum
  opt = tf.keras.optimizers.experimental.SGD(learning_rate, momentum=momentum)
  # loss function
  loss = tf.keras.losses.MeanSquaredError(y_train, y_pred)
  # initialize variables
  tf.compat.v1.global_variables_initializer()
  # train model
  for epoch in range(epochs):
    # both calculate the gradient and apply it to the variables
    opt.minimize(loss=loss, var_list=[w_pred, b_pred])
    print(f'Epoch {epoch}:', loss(y_true, y_pred).numpy())
  return w_pred
