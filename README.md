# MMEngine Demo Project

The aim of this repository is to demonstrate how to use the <a href="https://github.com/open-mmlab/mmengine" target="_blank">MMEngine</a> runtime library. Models can be easily configured using configuration files while MMEngine provides a runner that is responsible for training, evaluation, logging and much more.

<div style="display:flex; padding:1rem;">
  <div style="width:50%">
    <a href="" target="_blank"><img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" /></a>
  </div>
  <div style="width:50%">
    <a href="" target="_blank"><img src="https://user-images.githubusercontent.com/58739961/187154444-fce76639-ac8d-429b-9354-c6fac64b7ef8.jpg" /></a>
  </div>
</div>

## Setup thie project

```bash
# Create conda environment
$ conda create -y -n mmengine python==3.12 pip
$ conda activate mmengine

# Install PyTorch
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install OpenMMLab libraries
$ pip install openmim
$ mim install mmengine
$ mim install mmpretrain

# Install other requirements
$ pip install -r requirements.txt
```

## Download datasets

There are some default datasets that you can download for training. This can be done with the following commands:

```bash
# MNIST digits dataset
$ python scripts/datasets/generate_mnist.py

# Flowers102 Dataset
$ python scripts/datasets/generate_flowers102.py
```

## Train a simple CNN model

Within `./models/simple_cnn.py` you will find an implementation of a simple model that can classify images. The configuration for this model can be found under `./configs/simple_cnn.py`. The script for training, evaluation, testing, inference and export to the ONNX format can be found in the directory `./scripts/simple_cnn`.
