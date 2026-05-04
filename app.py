import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import streamlit as st

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
RESULTS_DIR = "./results"

EXPERIMENTS = [
    {
        "name":    "Exp1_ResNet50_FeatureExtract",
        "label":   "Model 1",
        "backbone":   "resnet50",
        "desc": """
**Backbone**: ResNet50  
**Strategy**: Feature Extraction (backbone 완전 동결)  
**Pretrained**: ✅ ImageNet  

backbone의 모든 가중치를 고정하고 마지막 분류 레이어만 학습.  
전이학습의 가장 보수적인 방법으로, 빠르게 학습되지만 성능 한계가 있음.
        """
    },
    {
        "name":    "Exp2_ResNet50_FullFinetune",
        "label":   "Model 2",
        "backbone":   "resnet50",
        "desc": """
**Backbone**: ResNet50  
**Strategy**: Full Fine-tuning (전체 레이어 학습)  
**Pretrained**: ✅ ImageNet  

ImageNet으로 사전학습된 가중치에서 시작하여 모든 레이어를 포켓몬 데이터로 재학습.  
Model 1과 backbone은 동일하지만 fine-tuning 전략이 달라 성능 차이를 비교할 수 있음.
        """
    },
    {
        "name":    "Exp3_EfficientNet_FullFinetune",
        "label":   "Model 3",
        "backbone":   "efficientnet_b0",
        "desc": """
**Backbone**: EfficientNetB0  
**Strategy**: Full Fine-tuning (전체 레이어 학습)  
**Pretrained**: ✅ ImageNet  

backbone을 ResNet50 → EfficientNetB0으로 교체. Model 2와 전략은 동일하여 순수하게 backbone 아키텍처의 차이를 비교할 수 있음.  
EfficientNet은 더 적은 파라미터로 높은 성능을 내도록 설계됨.
        """
    },
    {
        "name":    "Exp4_ResNet50_NoPretrain",
        "label":   "Model 4",
        "backbone":   "resnet50",
        "desc": """
**Backbone**: ResNet50  
**Strategy**: Full Fine-tuning (전체 레이어 학습)  
**Pretrained**: ❌ 없음 (random init)  

Model 2와 모든 조건이 동일하지만 pretrained weight 없이 완전 랜덤으로 초기화.  
전이학습의 효과를 직접 확인하기 위한 대조 실험.
        """
    },
]

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────
def build_model(backbone, num_classes):
    if backbone == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    else:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    return model

@st.cache_resource
def load_all():
    class_path = os.path.join(RESULTS_DIR, "class_names.json")
    if not os.path.exists(class_path):
        return None, None, f"class_names.json 없음 ({os.path.abspath(class_path)})"
    with open(class_path) as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    loaded = {}
    for cfg in EXPERIMENTS:
        weight_path = os.path.join(RESULTS_DIR, f"{cfg['name']}_best.pth")
        if not os.path.exists(weight_path):
            continue
        try:
            model = build_model(cfg["backbone"], num_classes)
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
            model.eval()
            loaded[cfg["name"]] = model
        except Exception as e:
            print(f"{cfg['name']} 로드 실패: {e}")
    return loaded, class_names, None

@torch.no_grad()
def predict(model, image, class_names, top_k=5):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    probs = torch.softmax(model(tensor), dim=1)[0]
    top = probs.topk(top_k)
    return [(class_names[i], float(p)) for i, p in zip(top.indices, top.values)]

# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="Pokemon Classifier", page_icon="⚡", layout="wide")
st.title("⚡ Pokemon Classifier")
st.caption("Transfer Learning으로 학습한 4가지 모델로 포켓몬을 분류합니다.")

models_dict, class_names, err = load_all()
if err:
    st.error(err)
    st.stop()
if not models_dict:
    st.error(f"모델 없음. results 폴더 확인: {os.path.abspath(RESULTS_DIR)}")
    if os.path.exists(RESULTS_DIR):
        st.write(os.listdir(RESULTS_DIR))
    st.stop()

# ──────────────────────────────────────────────
# 업로드 + Go 버튼
# ──────────────────────────────────────────────
uploaded = st.file_uploader("포켓몬 이미지 업로드", type=["jpg", "jpeg", "png", "webp"])
go = st.button("GO!", disabled=(uploaded is None))

# ──────────────────────────────────────────────
# 예측 결과
# ──────────────────────────────────────────────
if uploaded and go:
    image = Image.open(uploaded)

    st.subheader("🔍 예측 결과")
    col_img, col1, col2, col3, col4 = st.columns(5)

    with col_img:
        st.markdown("#### 입력 이미지")
        st.image(image, use_container_width=True)

    for col, cfg in zip([col1, col2, col3, col4], EXPERIMENTS):
        if cfg["name"] not in models_dict:
            continue
        results = predict(models_dict[cfg["name"]], image, class_names, top_k=5)
        with col:
            st.markdown(f"#### {cfg['label']}")
            st.success(f"**{results[0][0]}** ({results[0][1]*100:.1f}%)")
            for rank, (name, prob) in enumerate(results, 1):
                st.progress(prob, text=f"{rank}. {name} ({prob*100:.1f}%)")

# ──────────────────────────────────────────────
# 실험 설명 + 결과 테이블
# ──────────────────────────────────────────────
st.divider()
st.subheader("📋 모델 설명")

desc_cols = st.columns(4)
for i, cfg in enumerate(EXPERIMENTS):
    with desc_cols[i]:
        st.markdown(f"#### {cfg['label']}")
        st.markdown(cfg["desc"])

st.divider()
st.subheader("📊 실험 결과 비교")

csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")
if os.path.exists(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.insert(0, "모델", [cfg["label"] for cfg in EXPERIMENTS[:len(df)]])
    df = df.drop(columns=["experiment"], errors="ignore")
    st.dataframe(
        df.style.highlight_max(subset=["test_acc", "test_precision", "test_recall", "test_f1"], color="#d4edda"),
        use_container_width=True
    )
else:
    st.info("results_summary.csv 없음")