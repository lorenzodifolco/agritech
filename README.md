# Agritech Plant Health Classifier

![MLOps](https://img.shields.io/badge/MLOps-DVC%20%7C%20MLflow%20%7C%20Docker-green)
![Coverage](https://img.shields.io/badge/Coverage-91%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12-blue)

## Project Objectives
The primary goal of this project is to implement a complete **end-to-end MLOps system**, integrating advanced technologies to manage the entire lifecycle of a Machine Learning model.

We developed a plant health classifier designed to help farmers and agricultural engineers identify plant diseases early. The system analyzes leaf images and classifies them as "Healthy" or suffering from a specific pathology (e.g., *Apple Scab* or *Tomato Early Blight*), covering a total of **38 classes**.

---

## Supported Plant Species & Diseases
Our model is trained on the [**New Plant Diseases Dataset**](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset), allowing for the identification of health issues across various species:

* **Apple**: Scab, Black rot, Cedar apple rust, Healthy.
* **Blueberry**: Healthy.
* **Cherry**: Powdery mildew, Healthy.
* **Corn**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy.
* **Grape**: Black rot, Esca (Black Measles), Leaf blight (Isariopsis Leaf Spot), Healthy.
* **Orange**: Haunglongbing (Citrus greening).
* **Peach**: Bacterial spot, Healthy.
* **Pepper (Bell)**: Bacterial spot, Healthy.
* **Potato**: Early blight, Late blight, Healthy.
* **Raspberry**: Healthy.
* **Soybean**: Healthy.
* **Squash**: Powdery mildew.
* **Strawberry**: Leaf scorch, Healthy.
* **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy.

---

## Tech Stack
Based on the methodologies learned during the course, the project utilizes the following tools:

| Domain | Tools & Frameworks |
| :--- | :--- |
| **Data Versioning & Augmentation** | DVC, DagsHub, Albumentations |
| **Experiment Tracking & Tuning** | MLflow, OneCycleLR Scheduler |
| **Model Training & Framework** | PyTorch |
| **Testing & CI/CD** | pytest, GitHub Actions |
| **Deployment & UI** | Docker, MLServer, Streamlit |

---

## Repository Architecture
The project follows a modular structure to ensure scalability and separation of responsibilities:
* **`src/`**: Core of the system, including training, model definitions, and runtime logic.
* **`ui/`**: Independent user interface based on Streamlit.
* **`configs/`**: Centralized parameter management via `params.yml`.
* **`data/`**: Management of **87,000 images** (~2GB) via DVC metadata.
* **`tests/`**: Automated test suite to ensure production stability.

---

## Setup

### Download Data and Model
The dataset ("New Plant Diseases Dataset" from Kaggle) and model weights are managed via DVC to keep the Git repository lightweight:
```bash
pip install dvc
dvc pull
```

### Execution with Docker
The system utilizes containerization to ensure "run anywhere" consistency and easy deployment across different environments. By using Docker Compose, you can orchestrate both the inference engine and the user interface simultaneously.

* **MLServer**: Provides high-performance REST endpoints for real-time plant diagnosis.
* **Streamlit**: Offers a simple and intuitive web interface for uploading leaf photos from any device, providing specific disease classification and confidence scores instantly.

To build and run the services:
```bash
docker compose up --build
```

---

## Quality Assurance & CI/CD
Software quality and model reliability are core pillars of this project. We implement automated checks to ensure the dataset meets professional standards before training begins.

* **Validation & Drift**: The system performs automated checks for class imbalance and train-test leakage across the 87,000 images. It also identifies shifts in image quality or lighting that could compromise model reliability.
* **Automated Testing**: We use pytest to validate preprocessing logic and API response formats.
* **Continuous Integration**: GitHub Actions triggers the full test suite on every code push to prevent regressions and ensure only verified models reach production.
* **Experiment Tracking**: MLflow is utilized to log metrics (Accuracy, Loss), parameters, and the Confusion Matrix for all 38 classes, allowing for seamless comparison between different architectures.
* **Training Optimization**: We implement the OneCycleLR Scheduler to optimize the learning rate for efficient model convergence. The system also uses early stopping for poor-performing runs to optimize compute time.
* **Data Augmentation**: A stochastic pipeline using Albumentations (including Brightness, Rotation, and Gaussian Blur) prevents background overfitting and ensures robust generalization in varied lighting and field conditions.

---

## Useful Links
* [**GitHub Repository**](https://github.com/lorenzodifolco/agritech)
* [**DagsHub Project**](https://dagshub.com/lorenzodifolco00/agritech): Used for DVC remote storage and MLflow experiment tracking.

---

## System Demo
Watch the **Agritech Plant Health Classifier** in action. The video demonstrates the seamless upload process and the near-instant disease diagnosis.

![System Demo](ui/assets/demo_no.mp4)

---

## Team Members (Group 12)
This project was developed by Group 12 for the **Machine Learning and Data in Operation** course:
* **Gianluca Nogara**
* **Lorenzo Di Folco**
* **Kevin Hänggi**
* **Kai Erdin**