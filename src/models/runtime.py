import os
import torch
import numpy as np
from mlserver import MLModel
from mlserver.codecs import decode_args
from torchvision import transforms
from PIL import Image
import json

from src.models.model import PlantClassifier

# Resolve path relative to this file so it works regardless of working directory
_CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")


class PlantDiseaseRuntime(MLModel):

    async def load(self) -> bool:
        with open(_CLASS_NAMES_PATH) as f:
            self.class_names = json.load(f)

        self.model = PlantClassifier(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(torch.load(
            "model.pth", map_location=torch.device("cpu")
        ))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        # self.ready must be set explicitly — MLServer checks this property
        # to decide whether to accept inference requests
        self.ready = True
        return self.ready

    # @decode_args automatically decodes the V2 request payload into the annotated
    # Python type (np.ndarray) and encodes the return value back into V2 format
    @decode_args
    async def predict(self, payload: np.ndarray) -> np.ndarray:
        image = Image.fromarray(payload).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top4_confidences, top4_indices = torch.topk(probabilities, 4)

        top4 = [
            {"disease": self.class_names[idx.item()], "confidence": round(conf.item() * 100, 2)}
            for conf, idx in zip(top4_confidences[0], top4_indices[0])
        ]
        result = {"disease": top4[0]["disease"], "confidence": top4[0]["confidence"], "top3": top4[1:]}

        # Return as a 1-element string array; the client parses it with json.loads
        return np.array([json.dumps(result)])
