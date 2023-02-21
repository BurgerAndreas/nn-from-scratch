import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore tensorflow warnings

from helpers.get_data import create_linear_data
from models.regression.gd import GradientDescentMSELinearRegression
from models.regression.gd_tf import tf_gd_mse_lr, tf_gd
from models.regression.gd_pytorch import torch_gd_mse_nn, torch_gd_mse_lr
from models.regression.mlp_regression_scratch import scratch_mlp
from models.language.bigram_counter import test_counting_bigram_model
from models.language.bigram_nn import test_nn_bigram_model
from models.language.mlp_char import CharMLP

def main():
  print('-' * 80)
  # look at data
  # create_linear_data(dim=1, num_samples=1000, plot=True)

  # Stochastic Gradient Descent
  # GradientDescentMSELinearRegression()
  # tf_gd_mse_lr()
  # torch_gd_mse_nn()
  # torch_gd_mse_lr()

  # Multi-Layer Perceptron
  # implemented from scratch
  # trying to learn a step function
  # scratch_mlp()

  # test_counting_bigram_model()
  # test_nn_bigram_model()

  mlp = CharMLP()

if __name__ == '__main__':
  main()