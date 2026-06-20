# SehaTrack: AI-Powered Clinical Decision Support System

## Project Description

SehaTrack is an AI-powered clinical decision support system designed to assist healthcare professionals in analyzing different types of medical data within a single platform. The system integrates chest X-ray analysis, gastrointestinal image classification, speech-based symptom extraction, Explainable AI techniques, and patient record management to support faster and more informed clinical decision-making.

The platform provides an interactive web interface where healthcare professionals can upload medical images, analyze patient speech recordings, view AI-generated predictions, and access historical patient records through a centralized database.

---

## Team Members

| Name               | ID        | Program |
| ------------------ | --------- | ------- |
| Mennatullah Sharaf | 202201707 | DSAI    |
| Manar Mamdouh      | 202201881 | DSAI    |
| Rana Ahmed         | 202201899 | DSAI    |
| Rofida Ramadan     | 202201888 | DSAI    |

---

## Supervisor

**Dr. Yousry Abdelazem**

---

## Problem Statement

Healthcare professionals routinely work with different types of patient data, including medical images, speech recordings, and clinical observations. Analyzing these data sources separately can be time-consuming and may lead to fragmented workflows.

Most existing healthcare solutions focus on a single task, such as image analysis or symptom assessment, requiring clinicians to use multiple systems. This creates a need for an integrated platform capable of processing multiple forms of medical data while providing interpretable results that support clinical decision-making.

---

## Features

### Medical Image Analysis

* Chest X-ray disease detection.
* Gastrointestinal disease classification.
* Confidence score generation.

### Speech Analysis

* Medical speech transcription using Whisper.
* Symptom extraction using DistilBERT.
* Top-3 symptom prediction.

### Explainable AI

* Grad-CAM heatmaps for chest X-ray predictions.
* LIME explanations for gastrointestinal image classification.

### Patient Management

* Patient record management.
* Medical history tracking.
* Appointment management.
* Prescription management.

### Security and Administration

* Doctor authentication.
* Audit logging.
* Secure database storage.

### Dashboard and Reporting

* Interactive analytics dashboard.
* Clinical statistics and visualizations.
* Historical record retrieval.

---

## System Architecture

SehaTrack follows a modular architecture consisting of:

1. Authentication Module
2. Speech Analysis Module (Whisper + DistilBERT)
3. Chest X-ray Analysis Module (DenseNet121/CheXNet + Grad-CAM)
4. Gastrointestinal Analysis Module (EfficientNetB1 + LIME)
5. Visualization and Dashboard Module
6. Database Layer (SQLite)
7. Audit Logging Module

### Workflow

Doctor Login → Data Upload → Preprocessing → AI Model Inference → Prediction & Explainability → Dashboard Display → Database Storage

---

## Technologies Used

### Frontend

* Streamlit

### Backend

* Python

### Database

* SQLite

### AI / Machine Learning

* PyTorch
* Hugging Face Transformers
* Whisper
* DistilBERT
* DenseNet121 (CheXNet)
* EfficientNetB1

### Image Processing

* OpenCV
* PIL

### Explainable AI

* Grad-CAM
* LIME

### Data Processing

* Pandas
* NumPy
* Scikit-learn

### Visualization

* Matplotlib
* Plotly

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SehaTrack
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import streamlit, torch, transformers, whisper; print('Setup Successful')"
```

---

## Deployment Instructions

### Run Locally

```bash
streamlit run app.py
```

The application will be available through the local Streamlit URL displayed in the terminal.

### Cloud Deployment

The application can be deployed using Streamlit Community Cloud:

1. Push the project to GitHub.
2. Create a Streamlit Cloud account.
3. Connect the repository.
4. Select `app.py` as the entry point.
5. Configure any required secrets.
6. Deploy the application.

---

## Usage Guide

### Doctor Workflow

1. Log in using doctor credentials.
2. Select the desired analysis module.
3. Upload a chest X-ray image, GI image, or speech recording.
4. Wait for the AI model to process the input.
5. Review predictions, confidence scores, and explanations.
6. Save results to the patient record.

### Chest X-ray Analysis

1. Open the Chest X-ray module.
2. Upload an X-ray image.
3. View disease predictions.
4. Review Grad-CAM heatmaps.

### Gastrointestinal Analysis

1. Open the GI Analysis module.
2. Upload an endoscopic image.
3. View disease classification results.
4. Review LIME explanations.

### Speech Analysis

1. Upload or record patient speech.
2. Generate transcription.
3. View extracted symptoms.
4. Review the top predicted symptoms.

### Patient Records

1. Search for a patient.
2. Access previous visits and analyses.
3. Review historical diagnoses and reports.

---

## Future Enhancements

* Support for CT and MRI image analysis.
* Multilingual speech processing.
* Cloud-native deployment.
* Mobile application support.
* Integration with healthcare information systems.
* Multimodal AI fusion for improved diagnostic accuracy.

---
## Screenshots

## Login Page

![Login Page](screenshots/log-in.png)

## Chest X-Ray Analysis

![X-Ray Prediction](screenshots/xray_results.png)

## GI Analysis

![GI Prediction](screenshots/Gi_result.png)

## Speech Analysis

![Speech Prediction](screenshots/speech model.png)

## Dashboard

![Dashboard](screenshots/dashboard.png)

## Patient Records

![Patient Records](screenshots/Patient Records.png)
