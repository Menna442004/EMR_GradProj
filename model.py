"""
model.py — SehaTrack Pro
Three models — ALL lazy (nothing loads at import time):
  ① NLP  symptom classifier   → get_nlp_model()
  ② CheXNet X-ray classifier  → load_vision_engine()
  ③ GI  EfficientNetB1        → load_gi_engine()
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision.models import densenet121

# ── Silence transformers path warnings at the earliest possible point ─────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONFIG
# ══════════════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device

NLP_MODEL_PATH = r"C:\Users\RofaR\OneDrive\Desktop\ManarGP\model_only"
VISION_WEIGHTS = os.getenv("VISION_WEIGHTS", "best_chexnet_multimodal.pth")
# GI_WEIGHTS = r"C:\Users\RofaR\OneDrive\Desktop\ManarGP\Models\gi_model.keras"
GI_WEIGHTS = r"C:\Models\gi_model.keras"


# ══════════════════════════════════════════════════════════════════════════════
# ① NLP MODEL — lazy, cached in module-level dict
# ══════════════════════════════════════════════════════════════════════════════
# Nothing from transformers is imported here.
# The import happens inside get_nlp_model() on first call only.
_nlp_cache: dict = {}


def get_nlp_model():
    """
    Returns (tokenizer, model, id2label).
    Loads once on first call; subsequent calls return cached objects instantly.
    """
    if _nlp_cache:
        return _nlp_cache["tok"], _nlp_cache["model"], _nlp_cache["id2label"]

    # Deferred import — runs once, avoids 300+ __path__ warnings on every rerun
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok   = AutoTokenizer.from_pretrained(NLP_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL_PATH)
    model.to(device).eval()

    try:
        with open(os.path.join(NLP_MODEL_PATH, "id2label.json")) as f:
            id2label = json.load(f)
    except Exception:
        id2label = {str(i): f"Class {i}" for i in range(model.config.num_labels)}

    _nlp_cache["tok"]     = tok
    _nlp_cache["model"]   = model
    _nlp_cache["id2label"] = id2label
    return tok, model, id2label


def _get_probs(text: str):
    tok, model, _ = get_nlp_model()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=1)[0]


def predict(text: str):
    """Returns (label, confidence) for the top prediction."""
    probs   = _get_probs(text)
    pred_id = torch.argmax(probs).item()
    _, _, id2label = get_nlp_model()
    return id2label[str(pred_id)], float(probs[pred_id])


def predict_topk(text: str, k: int = None):
    """Returns predictions sorted by confidence. Each: {rank, label, score}."""
    probs      = _get_probs(text)
    _, _, id2label = get_nlp_model()
    sorted_ids = torch.argsort(probs, descending=True)
    if k is not None:
        sorted_ids = sorted_ids[:k]
    results = []
    for rank, idx in enumerate(sorted_ids, 1):
        score = float(probs[idx])
        if rank > 1 and score < 0.001:
            break
        results.append({"rank": rank, "label": id2label[str(idx.item())], "score": score})
    return results


def debug(text: str):
    tok, model, id2label = get_nlp_model()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    probs  = torch.nn.functional.softmax(logits, dim=0)
    print(f"\n── Input ──────────────────────────────")
    print(f"  Text: '{text}'")
    print(f"\n── Logits ─────────────────────────────")
    print(f"  min={logits.min():.4f}  max={logits.max():.4f}  std={logits.std():.4f}")
    print(f"\n── Top 10 predictions ─────────────────")
    for rank, idx in enumerate(torch.argsort(probs, descending=True)[:10], 1):
        i = idx.item()
        print(f"  {rank:>2}. {id2label.get(str(i), f'[id {i}]'):<40s}  {probs[i]*100:6.2f}%")
    print(f"\n── id2label check ─────────────────────")
    missing = [str(i) for i in range(model.config.num_labels) if str(i) not in id2label]
    print(f"  entries={len(id2label)}  num_labels={model.config.num_labels}")
    if missing:
        print(f"  ⚠️  Missing: {missing[:10]}")
    else:
        print(f"  ✅ All label IDs present")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ② VISION MODEL — DenseNet121 CheXNet Multimodal
# ══════════════════════════════════════════════════════════════════════════════
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

# ⚠️ Replace with actual values from notebook Cell 32 after training
OPTIMAL_THRESHOLDS = {
    "Atelectasis": 0.30, "Cardiomegaly": 0.35, "Effusion": 0.30,
    "Infiltration": 0.25, "Mass": 0.35, "Nodule": 0.35,
    "Pneumonia": 0.30, "Pneumothorax": 0.35, "Consolidation": 0.30,
    "Edema": 0.35, "Emphysema": 0.35, "Fibrosis": 0.35,
    "Pleural_Thickening": 0.30, "Hernia": 0.35,
}


def encode_meta(age: int, gender: str, view: str) -> torch.Tensor:
    """Exact encoding from notebook Cell 5: Male=0, Female=1 / PA=0, AP=1."""
    return torch.tensor(
        [[age / 100.0,
          0.0 if gender == "Male" else 1.0,
          0.0 if view   == "PA"   else 1.0]],
        dtype=torch.float32, device=DEVICE
    )


def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """CLAHE — matches notebook apply_clahe() exactly."""
    img   = np.array(pil_img.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))


class CheXNetMultimodal(nn.Module):
    def __init__(self, num_classes=14, meta_dim=3, dropout_rate=0.4):
        super().__init__()
        base = densenet121(weights=None)
        self.features = base.features
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        dense_out     = base.classifier.in_features  # 1024
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
        )
        fusion_dim = dense_out + 16
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fusion_dim), nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 512), nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )
        self._gradients = self._activations = None

    def _save_gradient(self, grad): self._gradients = grad

    def forward(self, img, meta):
        x = self.features(img)
        if x.requires_grad:
            x.register_hook(self._save_gradient)
        self._activations = x
        x = torch.flatten(self.avgpool(x), 1)
        return self.classifier(torch.cat([x, self.meta_branch(meta)], dim=1))


class GradCAM:
    def __init__(self, model: CheXNetMultimodal):
        self.model = model
        self._gradients = self._activations = None
        target = model.features.denseblock4.denselayer16.conv2
        target.register_forward_hook(lambda m,i,o: setattr(self,'_activations',o.detach()))
        target.register_full_backward_hook(lambda m,gi,go: setattr(self,'_gradients',go[0].detach()))

    def generate(self, img_tensor, meta_tensor, class_idx) -> np.ndarray:
        self.model.eval()
        logits = self.model(img_tensor.unsqueeze(0).to(DEVICE), meta_tensor.to(DEVICE))
        self.model.zero_grad()
        logits[0, class_idx].backward()
        cam = (self._gradients.mean(dim=[2,3], keepdim=True) * self._activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0: cam = cam / cam.max()
        return cv2.resize(cam.cpu().numpy(), (224, 224))

    @staticmethod
    def overlay(pil_img, cam, alpha=0.45) -> Image.Image:
        base  = np.array(pil_img.resize((224,224))).astype(np.float32) / 255.0
        heat  = cv2.cvtColor(
            cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0
        return Image.fromarray(((alpha*heat + (1-alpha)*base).clip(0,1)*255).astype(np.uint8))


def load_vision_engine(weights_path: str = VISION_WEIGHTS) -> CheXNetMultimodal:
    model = CheXNetMultimodal(num_classes=14)
    if os.path.exists(weights_path):
        ckpt  = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        state = (ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
                 ) if isinstance(ckpt, dict) else ckpt
        clean = {k.replace("module.","").replace("base_model.",""): v for k,v in state.items()}
        model.load_state_dict(clean, strict=False)
    else:
        print(f"[model.py] ⚠️  Vision weights not found at '{weights_path}'.")
    return model.to(DEVICE).eval()


# ══════════════════════════════════════════════════════════════════════════════
# ③ GI MODEL — EfficientNetB1 Kvasir Gastrointestinal Classifier
# ══════════════════════════════════════════════════════════════════════════════
import os
import numpy as np
import cv2
from PIL import Image

# load model path
GI_WEIGHTS = r"E:\SehaTrack2\Models\gi_model_clean.h5"

GI_CLASS_NAMES = [
    "dyed-lifted-polyps", "dyed-resection-margins", "esophagitis",
    "normal-cecum", "normal-pylorus", "normal-z-line",
    "polyps", "ulcerative-colitis",
]

_GI_RISK_MAP = {
    "polyps": "High", "ulcerative-colitis": "High",
    "esophagitis": "Medium",
    "dyed-lifted-polyps": "Low", "dyed-resection-margins": "Low",
    "normal-cecum": "Low", "normal-pylorus": "Low", "normal-z-line": "Low",
}


def get_gi_risk(label: str) -> str:
    return _GI_RISK_MAP.get(label, "Low")


def load_gi_engine():
    import tensorflow as tf
    if not os.path.exists(GI_WEIGHTS):
        print(f"[model.py] ⚠️  GI weights not found at '{GI_WEIGHTS}'.")
        return None

    # load model 
    return tf.keras.models.load_model(GI_WEIGHTS, compile=False)

def predict_gi(pil_img: Image.Image, gi_model) -> dict:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img       = pil_img.resize((224, 224)).convert("RGB")
    img_array = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    img_array = preprocess_input(img_array)

    preds     = gi_model.predict(img_array, verbose=0)
    pred_idx  = int(np.argmax(preds[0]))
    label     = GI_CLASS_NAMES[pred_idx]

    return {
        "label":      label,
        "confidence": float(preds[0][pred_idx]),
        "risk":       get_gi_risk(label),
        "all_probs":  {
            GI_CLASS_NAMES[i]: round(float(preds[0][i]), 4)
            for i in range(len(GI_CLASS_NAMES))
        },
    }

def gradcam_gi(pil_img: Image.Image, gi_model) -> np.ndarray:
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.applications.efficientnet import preprocess_input

    # 1) Preprocess image
    img = pil_img.resize((224, 224)).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    last_conv_layer = gi_model.get_layer("top_conv")

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in model.")

    grad_model = tf.keras.models.Model(
        inputs=gi_model.input,
        outputs=[last_conv_layer.output, gi_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # conv_outputs = tf.convert_to_tensor(conv_outputs)
        # predictions = tf.convert_to_tensor(predictions)

        # class_idx = int(np.argmax(predictions.numpy().flatten()))
        class_channel = tf.reduce_max(predictions, axis=1)

    grads = tape.gradient(class_channel, conv_outputs)

    if isinstance(conv_outputs, (list, tuple)):
        conv_outputs = conv_outputs[0]

    if isinstance(grads, (list, tuple)):
        grads = grads[0]

    conv_outputs = tf.convert_to_tensor(conv_outputs)
    grads = tf.convert_to_tensor(grads)

   
    conv_outputs = conv_outputs[0]
    grads = grads[0]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_gi_gradcam(pil_img: Image.Image, heatmap: np.ndarray,
                       alpha: float = 0.4) -> Image.Image:
    """Overlay Grad-CAM heatmap on original GI image."""
    img     = np.array(pil_img.resize((224, 224))).astype(np.float32)
    resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    blended = cv2.cvtColor(
        (colored * alpha + img).clip(0, 255).astype(np.uint8),
        cv2.COLOR_BGR2RGB
    )
    return Image.fromarray(blended)
