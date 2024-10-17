import click
from mmengine.runner import Runner
from mmengine.config import Config
import datasets
import models
from PIL import Image
import torch
from torchvision import transforms
from datasets.pipeline import build_pipeline
import onnxruntime as ort


@click.group()
def cli():
    """A CLI for training, validating, and testing a model using MMEngine."""
    pass


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to the configuration file.')
@click.option('--resume', type=click.Path(exists=True), required=False, help='Path to a checkpoint file to resume from.')
def train(config, resume):
    """Train the model."""
    cfg = Config.fromfile(config)
    runner = Runner.from_cfg(cfg)
    
    if resume:
        runner.load_checkpoint(resume)
    
    runner.train()


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to the configuration file.')
@click.option('--resume', type=click.Path(exists=True), required=True, help='Path to a checkpoint file to resume from.')
def val(config, resume):
    """Validate the model."""
    cfg = Config.fromfile(config)
    cfg.train_cfg = None  # Disable training
    cfg.train_dataloader = None
    cfg.optim_wrapper = None
    runner = Runner.from_cfg(cfg)
    
    if resume:
        runner.load_checkpoint(resume)
    
    runner.val()


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to the configuration file.')
@click.option('--resume', type=click.Path(exists=True), required=True, help='Path to a checkpoint file to resume from.')
def test(config, resume):
    """Test the model."""
    cfg = Config.fromfile(config)
    cfg.train_dataloader = None
    cfg.train_cfg = None  # Disable training
    cfg.val_dataloader = None
    cfg.val_cfg = None    # Disable validation
    cfg.val_evaluator = None    # Disable validation
    cfg.optim_wrapper = None
    runner = Runner.from_cfg(cfg)
    
    if resume:
        runner.load_checkpoint(resume)
    
    runner.test()


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to the configuration file.')
@click.option('--checkpoint', type=click.Path(exists=True), required=True, help='Path to the checkpoint file for the trained model.')
@click.option('--image-path', type=click.Path(exists=True), required=True, help='Path to the input image for inference.')
def inference(config, checkpoint, image_path):
    """Perform inference on a single image."""
    # Load the configuration and initialize the runner
    cfg = Config.fromfile(config)
    cfg.train_dataloader = None
    cfg.train_cfg = None  # Disable training
    cfg.val_dataloader = None
    cfg.val_cfg = None    # Disable validation
    cfg.val_evaluator = None    # Disable validation
    cfg.optim_wrapper = None
    runner = Runner.from_cfg(cfg)

    # Load the trained model checkpoint
    runner.load_checkpoint(checkpoint)
    model = runner.model.eval()  # Set model to evaluation mode

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image)

    # Perform inference
    with torch.no_grad():
        output = model(imgs=[image_tensor], labels=[-1], mode='predict')
    
    # Process and print the output
    print("OUTPUT", output)
    _, predicted_class = output[0]['pred_score'].max(0)
    print(f'Predicted class: {predicted_class.item()}')


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to the configuration file.')
@click.option('--checkpoint', type=click.Path(exists=True), required=True, help='Path to the checkpoint file for the trained model.')
@click.option('--output', type=click.Path(), required=True, help='Path to save the exported ONNX model.')
@click.option('--image-size', default=64, help='Input image size (default: 64)')
def export(config, checkpoint, output, image_size):
    """Export the model to ONNX format."""
    cfg = Config.fromfile(config)
    cfg.train_dataloader = None
    cfg.train_cfg = None  # Disable training
    cfg.val_dataloader = None
    cfg.val_cfg = None    # Disable validation
    cfg.val_evaluator = None    # Disable validation
    cfg.test_dataloader = None
    cfg.test_cfg = None    # Disable validation
    cfg.test_evaluator = None    # Disable validation
    cfg.optim_wrapper = None    # Disable validation
    runner = Runner.from_cfg(cfg)

    runner.load_checkpoint(checkpoint)
    model = runner.model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    print(f"Model exported to {output}")


@cli.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to the ONNX model file.')
@click.option('--image-path', type=click.Path(exists=True), required=True, help='Path to the input image for ONNX inference.')
def onnx_inference(model, image_path):
    """Perform inference using the ONNX model on a single image."""
    # Initialize the ONNX runtime session
    session = ort.InferenceSession(model)

    # Define the preprocessing for the image
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).numpy()  # Convert to NumPy for ONNX

    # Perform inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image_tensor})[0]
    
    # Process the output
    predicted_class = output.argmax(axis=1)[0]
    print(f'Predicted class from ONNX model: {predicted_class}')


if __name__ == '__main__':
    cli()