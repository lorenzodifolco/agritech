import torch
import pytest
import numpy as np
import os
from src.models.model import PlantClassifier
from src.data.dataset import PlantDiseaseDataset, get_train_transforms


# model test
def test_model_output_shape():
    """Controlla se il modello sputa fuori 38 classi"""
    model = PlantClassifier(num_classes=38)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    assert output.shape == (1, 38)


# --- TEST DEL DATASET ---
def test_dataset_logic(tmp_path):
    """Verifica che la logica del dataset funzioni (senza caricare 2GB)"""
    # Creiamo una mini-struttura di test temporanea
    d = tmp_path / "train" / "apple_scab"
    d.mkdir(parents=True)
    # Creiamo un'immagine finta (un file vuoto basta per testare l'esistenza)
    (d / "test_img.jpg").write_text("fake_image_data")

    # In una situazione reale qui useresti un'immagine vera,
    # ma pytest serve a testare se la classe Dataset inizializza bene i path
    ds = PlantDiseaseDataset(data_dir=str(tmp_path / "train"))
    assert len(ds.classes) == 1
    assert ds.classes[0] == "apple_scab"


# --- TEST DELLE TRASFORMAZIONI ---
def test_transforms():
    """Verifica che Albumentations faccia il suo lavoro"""
    transform = get_train_transforms()
    # Immagine dummy HWC (formato OpenCV)
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    transformed = transform(image=dummy_img)["image"]
    # Verifica che sia diventata un Tensor e sia 256x256
    assert torch.is_tensor(transformed)
    assert transformed.shape == (3, 256, 256)
