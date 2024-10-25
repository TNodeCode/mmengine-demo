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

## Train a model on the Flowers 102 dataset

After you have downloaded the Flowers102 dataset you can train a MobileNet model on this dataset with the following command:

```bash
$ python cli.py train --config ./configs/mobilenetv3_large/mobilenetv3_large_flowers102.py
```

The training logs and weight files can be found in `./work_dirs/mobilenetv3_large/flowers102` after training.
If you want to make changes to the training hyperparameters like the number of classes that your dataset has you can do this by copying the configuration file found in `./configs/mobilenetv3_large/mobilenetv3_large_flowers102.py` and adapt the parameters.

## Validate model

After training you can evaluate the model on the validation dataset using the following command:

```bash
$ python cli.py val --config ./configs/mobilenetv3_large/mobilenetv3_large_flowers102.py --resume work_dirs/mobilenetv3_large/flowers102/epoch_25.pth
```

You can also evaluate the model on the test dataset with this command:

```bash
$ python cli.py test --config ./configs/mobilenetv3_large/mobilenetv3_large_flowers102.py --resume work_dirs/mobilenetv3_large/flowers102/epoch_25.pth
```

## Performing Inference using PyTorch

If you want to perform inference on images using the trained model you can do this with the following command:

```bash
$ python cli.py inference \
  --config configs/mobilenetv3_large/mobilenetv3_large_flowers102.py \
  --checkpoint work_dirs/mobilenetv3_large/flowers102/epoch_22.pth \
  --image-path data/flowers102/test/0/0.png \
  --image-size 224
```

## Export the model to the ONNX format

When the model is trained it can be useful to export the model to a format that can be used without the PyTorch runtime. A popular format is ONNX, which can be read by many other runtimes like TensorFlow Lite or Tensor RT from NVidia. You can export your trained model using the following command:

```bash
$ python cli.py export \
  --config ./configs/mobilenetv3_large/mobilenetv3_large_flowers102.py \
  --checkpoint ./work_dirs/mobilenetv3_large/flowers102/epoch_22.pth \
  --output work_dirs/mobilenetv3_large/flowers102/model.onnx \
  --image-size 224
```

You should use the same image size that you have used during training to prevent the onnx model behaving different from the PyTorch model.

## Using ONNX model for inference

Within this repository we have included a command for testing your exported ONNX model using the ONNX runtime. You can test it with the following command:

```bash
$ python cli.py onnx-inference \
  --model ./work_dirs/mobilenetv3_large/flowers102/model.onnx \
  --image-path ./data/flowers102/test/0/0.png \
  --image-size 224
```