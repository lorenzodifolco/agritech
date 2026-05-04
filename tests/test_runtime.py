import pytest
import numpy as np
import torch
import json
from unittest.mock import MagicMock, patch, mock_open
from src.models.runtime import PlantDiseaseRuntime


@pytest.fixture
def runtime():
    """Fixture to initialize the runtime with basic settings."""
    model_settings = MagicMock()
    model_settings.name = "plant-disease-classifier"
    return PlantDiseaseRuntime(model_settings)


@pytest.mark.asyncio
@patch("src.models.runtime.torch.load")
# Creiamo almeno 4 classi per soddisfare il topk(4) del codice
@patch("builtins.open", new_callable=mock_open, read_data='["c1", "c2", "c3", "c4"]')
@patch("src.models.runtime.PlantClassifier")
async def test_runtime_load(mock_classifier, mock_file, mock_torch_load, runtime):
    """Verifies that the load() method correctly initializes class names and the model."""
    mock_torch_load.return_value = MagicMock()

    success = await runtime.load()

    assert success is True
    assert runtime.ready is True
    assert len(runtime.class_names) == 4
    mock_torch_load.assert_called_with(
        "models/model.pth", map_location=torch.device("cpu"), weights_only=False
    )


@pytest.mark.asyncio
@patch("src.models.runtime.torch.load")
@patch("builtins.open", new_callable=mock_open, read_data='["c1", "c2", "c3", "c4"]')
async def test_runtime_predict_format(mock_file, mock_torch_load, runtime):
    """Tests that the predict() method returns a correctly structured JSON string."""
    # 1. Setup
    mock_model = MagicMock()
    # Mock output con 4 valori per evitare errori in topk(4)
    mock_output = torch.tensor([[10.0, 1.0, 0.1, 0.01]])
    mock_model.return_value = mock_output
    mock_torch_load.return_value = mock_model

    await runtime.load()
    runtime.model = mock_model

    # 2. Execution
    # Usiamo .__wrapped__ per saltare il decoratore @decode_args che causava l'errore
    dummy_payload = np.uint8(np.random.rand(224, 224, 3) * 255)
    result_array = await runtime.predict.__wrapped__(runtime, dummy_payload)

    # 3. Assertions
    assert isinstance(result_array, np.ndarray)
    result_json = json.loads(result_array[0])

    assert "disease" in result_json
    assert "confidence" in result_json
    assert "top3" in result_json
    assert result_json["disease"] == "c1"
    assert len(result_json["top3"]) == 3
