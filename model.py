"""
model.py — SehaTrack Pro
NLP symptom classifier  +  CheXNet multimodal X-ray model
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision.models import densenet121
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ══════════════════════════════════════════════════════════════════════════════
# SHARED DEVICE
# ══════════════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device   # alias used by app.py


# ══════════════════════════════════════════════════════════════════════════════
# ① NLP MODEL  (original — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
model_path = r"C:\Users\RofaR\OneDrive\Desktop\ManarGP\model_only"

tokenizer = AutoTokenizer.from_pretrained(model_path)
nlp_model = AutoModelForSequenceClassification.from_pretrained(model_path)
nlp_model.to(device)
nlp_model.eval()

try:
    with open(f"{model_path}/id2label.json") as f:
        id2label = json.load(f)
except Exception:
    id2label = {str(i): f"Class {i}" for i in range(nlp_model.config.num_labels)}


def _get_probs(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = nlp_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    return probs


def predict(text: str):
    """Returns (label, confidence) for the top prediction. Backwards compatible."""
    probs   = _get_probs(text)
    pred_id = torch.argmax(probs).item()
    return id2label[str(pred_id)], float(probs[pred_id])


def predict_topk(text: str, k: int = None):
    """
    Returns all symptoms sorted by confidence descending.
    If k is given, returns only the top-k.
    Each item: {"rank": int, "label": str, "score": float}
    """
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
            "label": id2label[str(idx.item())],
            "score": score,
        })
    return results


def debug(text: str):
    """
    Run from terminal to diagnose prediction issues:
        python -c "from model import debug; debug('I have a headache')"
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = nlp_model(**inputs)
    logits = outputs.logits[0]
    probs  = torch.nn.functional.softmax(logits, dim=0)
    print(f"\n── Input ──────────────────────────────")
    print(f"  Text: '{text}'")
    print(f"\n── Logits (raw) ───────────────────────")
    print(f"  min={logits.min():.4f}  max={logits.max():.4f}  std={logits.std():.4f}")
    print(f"\n── Top 10 predictions ─────────────────")
    top_ids = torch.argsort(probs, descending=True)[:10]
    for rank, idx in enumerate(top_ids, 1):
        i     = idx.item()
        label = id2label.get(str(i), f"[MISSING id {i}]")
        print(f"  {rank:>2}. {label:<40s}  {probs[i]*100:6.2f}%")
    print(f"\n── id2label check ─────────────────────")
    print(f"  id2label entries : {len(id2label)}")
    print(f"  model num_labels : {nlp_model.config.num_labels}")
    missing = [str(i) for i in range(nlp_model.config.num_labels) if str(i) not in id2label]
    if missing:
        print(f"  ⚠️  Missing keys in id2label: {missing[:10]}")
    else:
        print(f"  ✅ All label IDs present in id2label.json")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# ② VISION MODEL — DenseNet121 CheXNet Multimodal
# ══════════════════════════════════════════════════════════════════════════════

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

# ⚠️  Replace with actual values from notebook Cell 32 after training completes
OPTIMAL_THRESHOLDS = {
    "Atelectasis":        0.30,
    "Cardiomegaly":       0.35,
    "Effusion":           0.30,
    "Infiltration":       0.25,
    "Mass":               0.35,
    "Nodule":             0.35,
    "Pneumonia":          0.30,
    "Pneumothorax":       0.35,
    "Consolidation":      0.30,
    "Edema":              0.35,
    "Emphysema":          0.35,
    "Fibrosis":           0.35,
    "Pleural_Thickening": 0.30,
    "Hernia":             0.35,
}

# Path to .pth file — set VISION_WEIGHTS env var or change the default string
VISION_WEIGHTS = os.getenv("VISION_WEIGHTS", "best_chexnet_multimodal.pth")


def encode_meta(age: int, gender: str, view: str) -> torch.Tensor:
    """
    Encode patient metadata exactly as done during training (notebook Cell 5).
    Gender : Male → 0.0,  Female → 1.0
    View   : PA   → 0.0,  AP     → 1.0
    Age    : raw_age / 100.0
    """
    age_norm  = age / 100.0
    gen_code  = 0.0 if gender == "Male" else 1.0
    view_code = 0.0 if view   == "PA"   else 1.0
    return torch.tensor([[age_norm, gen_code, view_code]],
                        dtype=torch.float32, device=DEVICE)


def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """CLAHE contrast enhancement — matches notebook apply_clahe() exactly."""
    img   = np.array(pil_img.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img)


class CheXNetMultimodal(nn.Module):
    """DenseNet121 backbone + 3-feature metadata branch (matches notebook Cell 21)."""

    def __init__(self, num_classes: int = 14, meta_dim: int = 3, dropout_rate: float = 0.4):
        super().__init__()
        base          = densenet121(weights=None)
        self.features = base.features
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        dense_out     = base.classifier.in_features   # 1024

        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),       nn.ReLU(),
        )
        fusion_dim = dense_out + 16   # 1040
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 512), nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )
        self._gradients  = None
        self._activations = None

    def _save_gradient(self, grad):
        self._gradients = grad

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = self.features(img)
        if x.requires_grad:          # guard: skip during eval / frozen phase
            x.register_hook(self._save_gradient)
        self._activations = x
        x     = self.avgpool(x)
        x     = torch.flatten(x, 1)
        m     = self.meta_branch(meta)
        fused = torch.cat([x, m], dim=1)
        return self.classifier(fused)


class GradCAM:
    """Gradient-weighted CAM on DenseNet121's last dense block."""

    def __init__(self, model: CheXNetMultimodal):
        self.model        = model
        self._gradients   = None
        self._activations = None
        target = model.features.denseblock4.denselayer16.conv2
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):   self._activations = o.detach()
    def _bwd(self, m, gi, go): self._gradients   = go[0].detach()

    def generate(self, img_tensor: torch.Tensor,
                 meta_tensor: torch.Tensor,
                 class_idx: int) -> np.ndarray:
        self.model.eval()
        img_t  = img_tensor.unsqueeze(0).to(DEVICE)
        meta_t = meta_tensor.to(DEVICE)
        logits = self.model(img_t, meta_t)
        self.model.zero_grad()
        logits[0, class_idx].backward()
        pooled = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam    = (pooled * self._activations).sum(dim=1).squeeze()
        cam    = F.relu(cam)
        cam    = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam.cpu().numpy(), (224, 224))

    @staticmethod
    def overlay(pil_img: Image.Image, cam: np.ndarray,
                alpha: float = 0.45) -> Image.Image:
        base  = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
        heat  = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blend = (alpha * heat + (1 - alpha) * base).clip(0, 1)
        return Image.fromarray((blend * 255).astype(np.uint8))


def load_vision_engine(weights_path: str = VISION_WEIGHTS) -> CheXNetMultimodal:
    """Load CheXNetMultimodal from .pth checkpoint."""
    model = CheXNetMultimodal(num_classes=14)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict):
            state = (ckpt.get("model_state_dict")
                     or ckpt.get("state_dict")
                     or ckpt)
        else:
            state = ckpt
        clean = {
            k.replace("module.", "").replace("base_model.", ""): v
            for k, v in state.items()
        }
        missing, unexpected = model.load_state_dict(clean, strict=False)
        if missing:
            print(f"[model.py] Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[model.py] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    else:
        print(f"[model.py] ⚠️  Weights not found at '{weights_path}'. Using random weights.")
    model.to(DEVICE).eval()
    return model
