# MMEngine Demo Project

The aim of this repository is to demonstrate how to use the <a href="https://github.com/open-mmlab/mmengine" target="_blank">MMEngine</a> runtime library. Models can be easily configured using configuration files while MMEngine provides a runner that is responsible for training, evaluation, logging and much more.

## Train a simple CNN model

Within `./models/simple_cnn.py` you will find an implementation of a simple model that can classify images. The configuration for this model can be found under `./configs/simple_cnn.py`. The script for training, evaluation, testing, inference and export to the ONNX format can be found in the directory `./scripts/simple_cnn`.
