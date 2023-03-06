import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore tensorflow warnings

from src.helpers.get_data import create_linear_data
from src.models.regression.gd import GradientDescentMSELinearRegression
from src.models.regression.gd_tf import tf_gd_mse_lr, tf_gd
from src.models.regression.gd_pytorch import torch_gd_mse_nn, torch_gd_mse_lr
from src.models.regression.mlp_regression_scratch import scratch_mlp
from src.models.language.bigram_counter import test_counting_bigram_model
from src.models.language.bigram_nn import test_nn_bigram_model
from src.models.language.mlp_scratch import CharMLP
from src.models.language.mlp_torch import CharMLP2
from src.models.language.wavenet import Wavenet
from src.models.language.bigram_torch import ShakespeareBigram
from src.models.language.transformer import ShakespeareLM


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

  # mlp = CharMLP()
  # improved version
  # mlp = CharMLP2()
  # mlp.train_and_test()

  # wavenet = convulutional neural network
  # model = Wavenet(len_context=8)

  # Shakespeare generative language model
  # model = ShakespeareBigram()
  model = ShakespeareLM(model_size='big')


if __name__ == '__main__':
  main()