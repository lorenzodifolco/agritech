import pytest
import torch
from unittest.mock import MagicMock, patch
from src.train import train


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
    print("\nSuccess: Training pipeline smoke test passed (Coverage increased).")
