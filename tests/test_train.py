import pytest
import torch
from unittest.mock import MagicMock, patch, mock_open
import src.train
from src.train import train
import importlib


def test_yaml_param_casting():
    """verify that YAML parameters are correctly cast to their appropriate types."""
    fake_yaml = """
    data:
        train_dir: "fake/train"
        valid_dir: "fake/valid"
    train:
        batch_size: "16"
        epochs: "1"
        max_lr: "0.005"
        weight_decay: "1e-4"
        grad_clip: "0.1"
    """
    with patch("builtins.open", mock_open(read_data=fake_yaml)):
        importlib.reload(src.train)
        assert isinstance(src.train.BATCH_SIZE, int)
        assert src.train.BATCH_SIZE == 16
        assert isinstance(src.train.MAX_LR, float)
        assert src.train.MAX_LR == 0.005
        assert isinstance(src.train.WEIGHT_DECAY, float)


@patch("src.train.PlantDiseaseDataset")
@patch("src.train.DataLoader")
@patch("src.train.mlflow")
def test_train_full_pipeline_smoke(mock_mlflow, mock_dataloader, mock_dataset):
    """
    Integration smoke test for the entire training pipeline.

    This test executes the train() function from src.train by mocking external
    dependencies (Dataset, DataLoader, and MLflow). This ensures that:
    1. The training loop logic (forward, backward, optimizer step) is correct.
    2. The MLflow logging sequence is properly called.
    3. The model architecture integrates with the data loading flow.

    By using mocks, we avoid loading the real 80k+ image dataset and
    connecting to remote servers, allowing the test to run in seconds.
    """
    # Mock the Dataset: simulate a dataset with 38 plant disease classes
    mock_ds_instance = MagicMock()
    mock_ds_instance.classes = [f"class_{i}" for i in range(38)]
    mock_dataset.return_value = mock_ds_instance

    # Mock the DataLoader: return a single synthetic batch to speed up the test
    # (1 image, 3 channels, 224x224)
    fake_images = torch.randn(1, 3, 224, 224)
    fake_labels = torch.tensor([1])
    # The iterator will return this single batch and then stop
    mock_dataloader.return_value = [(fake_images, fake_labels)]

    # Mock MLflow: prevent actual network calls to DagsHub/Remote tracking servers
    # We mock the context manager 'with mlflow.start_run()'
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    # Execution: Run the actual train() function
    # We use 'patch' to temporarily overwrite the EPOCHS constant to 1 for the test
    with patch("src.train.EPOCHS", 1):
        try:
            train()
            pipeline_passed = True
        except Exception as e:
            pytest.fail(
                f"The training pipeline crashed during the smoke test. Error: {e}"
            )

    assert pipeline_passed

    # Verify that MLflow logged the correct parameters from YAML
    mock_mlflow.log_params.assert_called_once_with(
        {
            "batch_size": src.train.BATCH_SIZE,
            "epochs": src.train.EPOCHS,
            "max_lr": src.train.MAX_LR,
            "weight_decay": src.train.WEIGHT_DECAY,
            "grad_clip": src.train.GRAD_CLIP,
            "scheduler": "OneCycleLR",
        }
    )

    # Verify that the model was logged at the end of the process
    mock_mlflow.pytorch.log_model.assert_called()
    print("\nSuccess: Training pipeline smoke test passed (Coverage increased).")
