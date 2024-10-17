# MMEngine Demo Project

The aim of this repository is to demonstrate how to use the <a href="https://github.com/open-mmlab/mmengine" target="_blank">MMEngine</a> runtime library. Models can be easily configured using configuration files while MMEngine provides a runner that is responsible for training, evaluation, logging and much more.

<div class="responsive-two-column-grid">
  <style>
    /* container */
    .responsive-two-column-grid {
        display:block;
    }

    /* columns */
    .responsive-two-column-grid > * {
        padding:1rem;
    }

    /* tablet breakpoint */
    @media (min-width:768px) {
        .responsive-two-column-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
        }
    }

  </style>
  <div>
    <a href="" target="_blank"><img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" /></a>
  </div>
  <div>
    <a href="" target="_blank"><img src="https://user-images.githubusercontent.com/58739961/187154444-fce76639-ac8d-429b-9354-c6fac64b7ef8.jpg" /></a>
  </div>
</div>

## Train a simple CNN model

Within `./models/simple_cnn.py` you will find an implementation of a simple model that can classify images. The configuration for this model can be found under `./configs/simple_cnn.py`. The script for training, evaluation, testing, inference and export to the ONNX format can be found in the directory `./scripts/simple_cnn`.
