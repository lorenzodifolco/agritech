import torch
import torch.nn as nn
import time
import pytest
from src.models.model import PlantClassifier

# Constants for testing
IMG_SIZE = 224
NUM_CLASSES = 38
LATENCY_THRESHOLD = 0.5  # Maximum allowed inference time in seconds


def fgsm_attack(image, epsilon, data_grad):
    """
    Performs the Fast Gradient Sign Method (FGSM) attack.

    This is an adversarial attack that creates an 'optical illusion' for the
    model by adding small perturbations to the input image based on the gradient
    of the loss.
    """
    # Collect the elements-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain the [0, 1] range
    return torch.clamp(perturbed_image, 0, 1)


@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_model_architecture_and_robustness():
    """
    Tests the ResNet18-based architecture and its robustness to adversarial noise.

    Verification steps:
    1. Instantiates the PlantClassifier to ensure the class is correctly defined.
    2. Performs a forward and backward pass to verify gradient flow.
    3. Executes an FGSM attack (epsilon=0.05) to check model stability.
    4. Asserts correct output shape and lack of numerical instabilities (NaNs).
    """

    # Set device to cpu for github actions
    device = torch.device("cpu")

    # Initialize model in evaluation mode
    model = PlantClassifier(num_classes=NUM_CLASSES).to(device)
    model.eval()

    # Create synthetic input with gradient tracking enabled for the attack
    input_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, requires_grad=True)
    target = torch.tensor([0])  # Dummy label ('Apple_Healthy')

    # Forward pass and Loss calculation
    output = model(input_tensor)
    loss = nn.CrossEntropyLoss()(output, target)

    # Backward pass to compute gradients relative to the input image
    model.zero_grad()
    loss.backward()

    # Apply FGSM attack (Small perturbation to test robustness)
    epsilon = 0.05
    perturbed_data = fgsm_attack(input_tensor, epsilon, input_tensor.grad.data)

    # Inference on the "attacked" (perturbed) image
    output_perturbed = model(perturbed_data)

    # Assertions to verify model architecture and robustness
    # A. Check output shape consistency
    assert output_perturbed.shape == (
        1,
        NUM_CLASSES,
    ), f"Shape mismatch: expected {NUM_CLASSES} classes, got {output_perturbed.shape[1]}"

    # B. Check for numerical stability (No NaNs in output)
    assert not torch.isnan(
        output_perturbed
    ).any(), "Model produced NaN outputs during attack!"

    # C. Verify gradient flow
    assert (
        input_tensor.grad is not None
    ), "Gradients did not flow back to the input tensor!"

    print(
        f"Success: Architecture and FGSM robustness verified for {NUM_CLASSES} classes."
    )


def test_inference_latency():
    """
    Benchmarks the model's inference speed on CPU.

    In an Agritech context (e.g., real-time disease detection on drones/tractors),
    low latency is critical. This test ensures the model responds within 500ms.
    """
    model = PlantClassifier(num_classes=NUM_CLASSES)
    model.eval()

    # Generate synthetic input
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    end_time = time.time()

    latency = end_time - start_time

    # Assert that the inference is fast enough for a standard CPU environment
    assert latency < LATENCY_THRESHOLD, f"Latency too high: {latency:.4f} seconds!"
    print(f"Success: Inference latency verified ({latency:.4f}s).")
