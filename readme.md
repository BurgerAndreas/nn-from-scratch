# Neural Networks from Scratch

Understanding how Neural Networks work by implementing them from from the bottom up.
Building up towards Generative Language Models.

Inspired by Andrej Karpathy's [makemore series](https://github.com/karpathy/makemore)

## Usage

```bash
docker build -t ml-image -f dockerfile .
docker images
# -it is short for --interactive + --tty
# will open a shell in the container (/bin/bash)
docker run -it --name ml-container ml-image
docker rm -f ml-container
```

ToDo: bind local directory to docker container

```bash
# -v <source>:<target>
# docker run --name docker-ml-model -v $(pwd)/nn-from-scratch/saved_models:/nn-from-scratch/saved_models
```

## Models

[] to be done

MLinPhysics - homework for the course ML in Physics at the University of Munich (LMU)

## Datasets

[LinkedIn Data Analyst jobs listings](https://www.kaggle.com/datasets/cedricaubin/linkedin-data-analyst-jobs-listings)

[Electronics reviews](https://data.world/datafiniti/amazon-and-best-buy-electronics)

[Names](https://www.ssa.gov/oact/babynames/)

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)