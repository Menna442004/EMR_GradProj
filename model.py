# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import json

# model_path = r"C:\Users\mamdo\OneDrive\Documents\grad\model_only"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# try:
#     with open(f"{model_path}/id2label.json") as f:
#         id2label = json.load(f)
# except:
#     id2label = {str(i): f"Class {i}" for i in range(model.config.num_labels)}


# def _get_probs(text: str):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
#     if "token_type_ids" in inputs:
#         inputs.pop("token_type_ids")
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
#     return probs


# def predict(text: str):
#     """Returns (label, confidence) for the top prediction. Backwards compatible."""
#     probs = _get_probs(text)
#     pred_id = torch.argmax(probs).item()
#     return id2label[str(pred_id)], float(probs[pred_id])


# def predict_topk(text: str, k: int = None):
#     """
#     Returns all symptoms sorted by confidence descending.
#     If k is given, returns only the top-k.
#     Each item: {"rank": int, "label": str, "score": float}
#     """
#     probs = _get_probs(text)
#     sorted_ids = torch.argsort(probs, descending=True)

#     if k is not None:
#         sorted_ids = sorted_ids[:k]

#     results = []
#     for rank, idx in enumerate(sorted_ids, 1):
#         score = float(probs[idx])
#         if rank > 1 and score < 0.001:
#             break
#         results.append({
#             "rank":  rank,
#             "label": id2label[str(idx.item())],
#             "score": score,
#         })

#     return results


# def debug(text: str):
#     """
#     Run from terminal to diagnose prediction issues:
#         python -c "from model import debug; debug('I have a headache')"
#     """
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
#     if "token_type_ids" in inputs:
#         inputs.pop("token_type_ids")
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits[0]
#     probs  = torch.nn.functional.softmax(logits, dim=0)

#     print(f"\n── Input ──────────────────────────────")
#     print(f"  Text: '{text}'")
#     print(f"\n── Logits (raw) ───────────────────────")
#     print(f"  min={logits.min():.4f}  max={logits.max():.4f}  std={logits.std():.4f}")

#     print(f"\n── Top 10 predictions ─────────────────")
#     top_ids = torch.argsort(probs, descending=True)[:10]
#     for rank, idx in enumerate(top_ids, 1):
#         i = idx.item()
#         label = id2label.get(str(i), f"[MISSING id {i}]")
#         print(f"  {rank:>2}. {label:<40s}  {probs[i]*100:6.2f}%")

#     print(f"\n── id2label check ─────────────────────")
#     print(f"  id2label entries : {len(id2label)}")
#     print(f"  model num_labels : {model.config.num_labels}")
#     missing = [str(i) for i in range(model.config.num_labels) if str(i) not in id2label]
#     if missing:
#         print(f"  ⚠️  Missing keys in id2label: {missing[:10]}")
#     else:
#         print(f"  ✅ All label IDs present in id2label.json")
#     print()


"""
model.py — SehaTrack Pro
Unified inference engine:
  ① NLP symptom classifier  (HuggingFace Transformers)
  ② CheXNet multimodal X-ray engine (DenseNet-121 + metadata branch)
  ③ Kvasir GI endoscopy engine  (EfficientNetB1 via TF/Keras)
  ④ Grad-CAM++ explainability for CheXNet
  ⑤ LIME explainability for Kvasir

All paths are relative — place your weight files next to this script:
    model_only/           ← HuggingFace NLP model directory
    best_chexnet_multimodal.pth
    gi_model_clean.h5
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import densenet121
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Lazy TF import so the app still loads if TF is not installed ──────────────
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    tf = None
    _TF_AVAILABLE = False

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# SHARED PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))

NLP_MODEL_PATH   = os.path.join(_HERE, "model_only")
VISION_WEIGHTS   = os.path.join(_HERE, "best_chexnet_multimodal.pth")
KVASIR_MODEL_PATH = os.path.join(_HERE, "gi_model_clean.h5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device

GI_CLASSES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

OPTIMAL_THRESHOLDS = {
    "Atelectasis": 0.38, "Cardiomegaly": 0.42, "Effusion": 0.40,
    "Infiltration": 0.35, "Mass": 0.30, "Nodule": 0.28,
    "Pneumonia": 0.45, "Pneumothorax": 0.33, "Consolidation": 0.37,
    "Edema": 0.41, "Emphysema": 0.29, "Fibrosis": 0.25,
    "Pleural_Thickening": 0.32, "Hernia": 0.20,
}


# ══════════════════════════════════════════════════════════════════════════════
# ① NLP SYMPTOM ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def _load_nlp():
    if not os.path.isdir(NLP_MODEL_PATH):
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained(NLP_MODEL_PATH)
        mdl = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL_PATH)
        mdl.to(device)
        mdl.eval()

        # Load id2label from JSON if not baked into config
        id2label_path = os.path.join(NLP_MODEL_PATH, "id2label.json")
        if os.path.isfile(id2label_path):
            with open(id2label_path) as f:
                extra = json.load(f)
            if not mdl.config.id2label:
                mdl.config.id2label = {int(k): v for k, v in extra.items()}

        return tok, mdl
    except Exception as e:
        print(f"[NLP] Failed to load: {e}")
        return None, None


_tokenizer, _nlp_model = _load_nlp()


def _get_probs(text: str) -> torch.Tensor:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _nlp_model(**inputs).logits
    return F.softmax(logits, dim=1)[0]


def predict(text: str) -> Tuple[str, float]:
    """Return (top_label, confidence) — backwards-compatible with original app.py."""
    if _nlp_model is None or not text.strip():
        return "Unknown Symptom", 0.0
    probs   = _get_probs(text)
    pred_id = torch.argmax(probs).item()
    label   = _nlp_model.config.id2label.get(pred_id, f"Class {pred_id}")
    return str(label), float(probs[pred_id])


def predict_topk(text: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return all symptoms sorted by confidence (desc).
    If k is given, return only top-k.
    Each item: {"rank": int, "label": str, "score": float}
    """
    if _nlp_model is None or not text.strip():
        return [{"rank": 1, "label": "Unknown Symptom", "score": 0.0}]

    probs      = _get_probs(text)
    sorted_ids = torch.argsort(probs, descending=True)
    if k is not None:
        sorted_ids = sorted_ids[:k]

    results = []
    for rank, idx in enumerate(sorted_ids, 1):
        score = float(probs[idx])
        if rank > 1 and score < 0.001:
            break
        results.append({
            "rank":  rank,
            "label": str(_nlp_model.config.id2label.get(idx.item(), f"Class {idx.item()}")),
            "score": score,
        })
    return results


def debug_nlp(text: str) -> None:
    """CLI helper: python -c "from model import debug_nlp; debug_nlp('I have a headache')" """
    if _nlp_model is None:
        print("NLP model not loaded.")
        return
    probs   = _get_probs(text)
    top_ids = torch.argsort(probs, descending=True)[:10]
    print(f"\n── Input: '{text}'")
    print(f"── Top 10 predictions ─────────────────")
    for rank, idx in enumerate(top_ids, 1):
        i     = idx.item()
        label = _nlp_model.config.id2label.get(i, f"[MISSING {i}]")
        print(f"  {rank:>2}. {label:<40s}  {probs[i]*100:6.2f}%")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ② CHEXNET MULTIMODAL X-RAY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def encode_meta(age: Any, gender: Any, view_pos: Any) -> torch.Tensor:
    """Encode patient metadata into a 3-value tensor for the CheXNet meta branch."""
    try:
        a = max(0.0, min(float(age), 120.0)) / 120.0
    except Exception:
        a = 0.0
    g = 1.0 if str(gender).lower().strip() == "female" else 0.0
    v = 1.0 if "ap" in str(view_pos).lower() else 0.0
    return torch.tensor([[a, g, v]], dtype=torch.float32)


class CheXNetMultimodal(nn.Module):
    def __init__(self, num_classes: int = 14, meta_dim: int = 3, dropout_rate: float = 0.4):
        super().__init__()
        base            = densenet121(weights=None)
        self.features   = base.features
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        dense_out       = base.classifier.in_features

        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
        )

        fusion = dense_out + 16
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fusion),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        feats = F.relu(self.features(img), inplace=False)
        x     = torch.flatten(self.avgpool(feats), 1)
        return self.classifier(torch.cat([x, self.meta_branch(meta)], dim=1))


def load_vision_engine(weights_path: str = VISION_WEIGHTS) -> CheXNetMultimodal:
    model = CheXNetMultimodal(num_classes=len(DISEASE_LABELS))
    if os.path.isfile(weights_path):
        try:
            ckpt  = torch.load(weights_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            if isinstance(state, dict):
                clean = {
                    k.replace("module.", "").replace("base_model.", ""): v
                    for k, v in state.items()
                }
                model.load_state_dict(clean, strict=False)
        except Exception as e:
            print(f"[CheXNet] Error loading weights: {e}")
    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ③ GRAD-CAM++ EXPLAINABILITY FOR CHEXNET
# ══════════════════════════════════════════════════════════════════════════════
class GradCAMPlusPlus:
    def __init__(self, model: CheXNetMultimodal):
        self.model       = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Hook onto the final dense block for stable spatial resolution
        target = (
            model.features.denseblock4
            if hasattr(model.features, "denseblock4")
            else list(model.features.children())[-2]
        )
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _m, _i, output: torch.Tensor) -> None:
        self.activations = output.detach().clone()

    def _save_gradient(self, _m, _gi, grad_output: Tuple) -> None:
        self.gradients = grad_output[0].detach().clone()

    def generate(
        self,
        img_tensor:  torch.Tensor,
        meta_tensor: torch.Tensor,
        category_idx: int,
    ) -> Optional[np.ndarray]:
        self.model.zero_grad()
        img  = img_tensor.detach().clone().requires_grad_(True)
        meta = meta_tensor.detach().clone()

        with torch.enable_grad():
            output = self.model(img, meta)
            category_idx = int(category_idx)
            if not (0 <= category_idx < output.shape[1]):
                category_idx = int(torch.argmax(output, dim=1).item())
            output[0, category_idx].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        g2 = self.gradients ** 2
        g3 = self.gradients ** 3
        gsum = self.activations.sum(dim=(2, 3), keepdim=True)

        denom   = 2.0 * g2 + gsum * g3
        denom   = torch.where(denom != 0, denom, torch.ones_like(denom))
        alphas  = g2 / denom
        weights = (alphas * F.relu(self.gradients)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam).detach().cpu().numpy()

        max_val = cam.max()
        cam = cam / max_val if max_val > 1e-5 else np.zeros_like(cam)
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (3, 3), 0)
        return cam

    # alias for callers that use generate_heatmap
    def generate_heatmap(self, img_tensor, meta_tensor, category_idx):
        return self.generate(img_tensor, meta_tensor, category_idx)

    @staticmethod
    def overlay(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
        base = np.array(pil_img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
        heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blend = (alpha * heat + (1.0 - alpha) * base).clip(0, 1)
        return Image.fromarray((blend * 255).astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# ④ KVASIR GI ENDOSCOPY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def load_kvasir_engine():
    if not _TF_AVAILABLE:
        return None
    if os.path.isfile(KVASIR_MODEL_PATH):
        try:
            return tf.keras.models.load_model(KVASIR_MODEL_PATH)
        except Exception as e:
            print(f"[Kvasir] Failed to load saved model: {e}")

    # Fallback: build untrained architecture
    base = tf.keras.applications.EfficientNetB1(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    model = tf.keras.models.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(GI_CLASSES), activation="softmax"),
    ])
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ LIME EXPLAINABILITY FOR KVASIR
# ══════════════════════════════════════════════════════════════════════════════
def run_kvasir_lime_explanation(
    pil_image: Image.Image,
    trained_model,
    num_samples: int = 500,
) -> Tuple[str, float, List[Dict], Image.Image]:
    """
    Returns (class_name, confidence, chart_data, lime_boundary_image).
    Falls back gracefully if LIME is not installed.
    """
    img_array = np.array(pil_image.convert("RGB").resize((224, 224))).astype(np.float32)

    # Get base prediction
    batch      = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(img_array, 0)
    )
    preds      = trained_model.predict(batch, verbose=0)[0]
    top_idx    = int(np.argmax(preds))
    class_name = GI_CLASSES[top_idx]
    confidence = float(preds[top_idx])

    if not _LIME_AVAILABLE:
        return class_name, confidence, [], pil_image.resize((224, 224))

    def classifier_fn(images: np.ndarray) -> np.ndarray:
        proc = tf.keras.applications.efficientnet.preprocess_input(
            images.astype(np.float32)
        )
        return trained_model.predict(proc, verbose=0)

    explainer   = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array, classifier_fn,
        top_labels=1, hide_color=0, num_samples=num_samples,
    )

    dict_weights   = explanation.local_exp.get(top_idx, [])
    sorted_weights = sorted(dict_weights, key=lambda x: abs(x[1]), reverse=True)[:6]

    temp, mask = explanation.get_image_and_mask(
        top_idx, positive_only=True, num_features=6, hide_rest=False
    )
    boundary_img = Image.fromarray(
        (mark_boundaries(temp / 255.0, mask) * 255).astype(np.uint8)
    )

    chart_data = [
        {
            "Feature/Superpixel Segment": f"Segment Region ID #{seg_id}",
            "LIME Attribution Weight": float(w),
        }
        for seg_id, w in sorted_weights
    ]

    return class_name, confidence, chart_data, boundary_img
