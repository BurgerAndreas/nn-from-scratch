FROM python:3.10-slim
# FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /nn-from-scratch

# Already installed in python:3.10-slim
# RUN apt-get update && apt-get install -y 
# RUN pip install --upgrade pip

# generated from conda with
# pip list --format=freeze > requirements.txt
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the source code and data into the container
COPY ./nn-from-scratch/ .

# Directory on the host machine that is mounted into the container at runtime
# alternatively bind mount with: docker run -v <source local>:<target container>
# VOLUME /home/<user>/Coding/nn-from-scratch/

# Set environment variables for paths
# ENV DIR_MODEL_SAVE=/home/<user>/Coding/nn-from-scratch/nn-from-scratch/models/saved_models/
# DIR_MODEL_SAVE = os.environ["DIR_MODEL_SAVE"] # in script.py

# RUN python3 main.py
CMD ["python","-u","main.py"]