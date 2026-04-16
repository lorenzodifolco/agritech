import torch
import time
import pytest
from src.models.model import PlantClassifier

# Threshold for acceptable inference time on a standard CPU
LATENCY_THRESHOLD_SECONDS = 0.5


def test_inference_latency():
    """
    Benchmarks the model's inference speed to ensure production readiness.

    This test verifies that:
    1. A single forward pass on CPU completes within 500ms.
    2. The model operates efficiently without unnecessary gradient overhead.
    """
    # Initialize the model with the standard 38 classes
    model = PlantClassifier(num_classes=38)
    model.eval()

    # Create a synthetic input tensor (Batch 1, 3 Channels, 224x224)
    # Using torch.randn simulates a preprocessed image
    dummy_input = torch.randn(1, 3, 224, 224)

    # Performance measurement
    # We use torch.no_grad() to disable the gradient engine,
    # which mimics actual production/inference behavior and saves memory.
    start_time = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    end_time = time.time()

    latency = end_time - start_time

    # Assertion: Inference must be faster than the defined threshold
    assert (
        latency < LATENCY_THRESHOLD_SECONDS
    ), f"Latency Failure: Model took {latency:.4f} seconds, which exceeds the {LATENCY_THRESHOLD_SECONDS}s limit!"

    print(f"\nSuccess: Inference latency verified at {latency:.4f}s.")
