# SehaTrack

AI-powered clinical platform — chest X-ray diagnosis + symptom analysis.

## Models Required (not included)

- `best_chexnet_multimodal.pth` — DenseNet121 weights (train from Kaggle notebook)
- `model_only/` — NLP symptom classifier (place in root directory)

## Run

pip install -r requirements.txt
streamlit run app.py
