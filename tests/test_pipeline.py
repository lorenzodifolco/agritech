import torch
import pytest
import numpy as np
import os
from src.models.model import PlantClassifier
from src.data.dataset import PlantDiseaseDataset, get_train_transforms


# model test
def test_model_output_shape():
    """Check the model produces the correct output shape for a dummy input"""
    model = PlantClassifier(num_classes=38)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    assert output.shape == (1, 38)


# dataset test
def test_dataset_logic(tmp_path):
    """Check that the dataset logic works (without loading 2GB)"""
    # generate a temporary directory structure with a single class and a dummy image
    d = tmp_path / "train" / "apple_scab"
    d.mkdir(parents=True)
    # generate a dummy image file (not a real image, just to test the dataset logic)
    (d / "test_img.jpg").write_text("fake_image_data")
    ds = PlantDiseaseDataset(data_dir=str(tmp_path / "train"))
    assert len(ds.classes) == 1
    assert ds.classes[0] == "apple_scab"


# transforms test
def test_transforms():
    """Check that Albumentations does its job and produces a tensor of the correct shape"""
    transform = get_train_transforms()
    # Dummy image in OpenCV format
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    transformed = transform(image=dummy_img)["image"]
    # verify that it is a tensor and has the correct shape (256x256)
    assert torch.is_tensor(transformed)
    assert transformed.shape == (3, 256, 256)
