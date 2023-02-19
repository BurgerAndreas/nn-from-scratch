from models.bigram_counter import test_counting_bigram_model
from models.bigram_nn import test_nn_bigram_model
from models.sgd import GradientDescentMSELinearRegression
from models.sgd_tf import tf_sgd_mse_lr, tf_sgd
from models.sgd_pytorch import torch_sgd_mse_nn, torch_sgd_mse_lr

def main():
  # test_counting_bigram_model()
  # test_nn_bigram_model()

  # Stochastic Gradient Descent
  # GradientDescentMSELinearRegression()
  # tf_sgd_mse_lr()
  # tf_sgd() # doesn't work
  # torch_sgd_mse_nn()
  torch_sgd_mse_lr()



if __name__ == '__main__':
  main()