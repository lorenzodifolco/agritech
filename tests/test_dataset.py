import torch
import numpy as np
from PIL import Image
import pytest


def test_preprocessing_output_shape():
    """
    Validates the image preprocessing pipeline used in production (Streamlit/MLServer).

    This test ensures that:
    1. The image is correctly resized to the required 224x224 dimensions.
    2. The dimensions are transposed from HWC (Height, Width, Channels) to
       CHW (Channels, Height, Width) to satisfy PyTorch's input requirements.
    3. The pixel values remain within the valid [0, 255] range.
    """
    # Simulate loading a random user-uploaded image (800x1000 pixels)
    random_img = np.uint8(np.random.rand(800, 1000, 3) * 255)
    img = Image.fromarray(random_img)

    # Mimic the transformation logic applied during inference
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).transpose(2, 0, 1)  # HWC -> CHW conversion

    # Assertions to verify data integrity
    assert img_array.shape == (
        3,
        224,
        224,
    ), "Error: Resize did not produce the expected CHW shape!"
    assert (
        img_array.max() <= 255
    ), "Error: Pixel values exceed the valid range [0, 255]!"
    print("Success: Preprocessing output shape and value range verified.")


def test_black_image_inference():
    """
    Stress test for model numerical stability using a null (all-black) input.

    This test checks if the model can handle a zero-input tensor without:
    1. Raising runtime exceptions or crashing.
    2. Producing NaN (Not a Number) values in the output, which could
       indicate mathematical instabilities in normalization layers.
    """
    from src.models.model import PlantClassifier

    # Initialize model in evaluation mode
    model = PlantClassifier(num_classes=38)
    model.eval()

    # Generate a dummy black image tensor (Batch Size 1, 3 Channels, 224x224)
    black_img = torch.zeros(1, 3, 224, 224)

    try:
        with torch.no_grad():
            output = model(black_img)

        # Verify that output is numerically stable
        assert not torch.isnan(
            output
        ).any(), "Error: Model produced NaN outputs for a black image!"
        print("Success: Model handled black image input without numerical instability.")

    except Exception as e:
        pytest.fail(
            f"Test Failed: The model crashed during inference with a null input. Error: {e}"
        )
