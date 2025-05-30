import streamlit as st
import cv2
import os
import torch
import timm
import tempfile
import numpy as np
from torchvision import transforms, models
from PIL import Image
import requests
import os

@st.cache_resource
def load_model(model_name):
    url_map = {
        "cvit": "https://huggingface.co/Stella1301/Sample-Skripsi/resolve/main/best_cvit.pth",
        "vit": "https://huggingface.co/Stella1301/Sample-Skripsi/resolve/main/best_vit.pth",
        "mobilenetv3": "https://huggingface.co/Stella1301/Sample-Skripsi/resolve/main/best_mobilenetv3.pth"
    }

    filename_map = {
        "cvit": "best_cvit.pth",
        "vit": "best_vit.pth",
        "mobilenetv3": "best_mobilenetv3.pth"
    }

    model_url = url_map[model_name]
    filename = filename_map[model_name]

    # Download jika file belum ada
    if not os.path.exists(filename) or os.path.getsize(filename) < 100_000:
        response = requests.get(model_url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)

    # Load model
    if model_name == "mobilenetv3":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
        model.load_state_dict(torch.load(filename, map_location='cpu'))
    elif model_name == "vit":
        model = models.vit_b_16(weights=None)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, 2)
        model.load_state_dict(torch.load(filename, map_location='cpu'))
    elif model_name == "cvit":
        model = timm.create_model("cait_s24_224", pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(filename, map_location='cpu'))

    model.eval()
    return model


# Setup transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@st.cache_resource
def load_model(model_name):
    if model_name == "mobilenetv3":
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
        model.load_state_dict(torch.load("best_mobilenetv3.pth", map_location='cpu'))
    elif model_name == "vit":
        model = models.vit_b_16(pretrained=False)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, 2)
        model.load_state_dict(torch.load("best_vit.pth", map_location='cpu'))
    elif model_name == "cvit":
        model = timm.create_model("cait_s24_224", pretrained=False, num_classes=2)
        model.load_state_dict(torch.load("best_cvit.pth", map_location='cpu'))
    else:
        raise ValueError("Model tidak dikenali.")
    
    model.eval()
    return model

def extract_faces(video_path, max_frames=30):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    faces = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in dets:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            faces.append(face)
            count += 1
            break  # Ambil 1 wajah per frame
    cap.release()
    return faces

def compute_farneback(faces):
    flows = []
    prev_gray = None
    for face in faces:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros_like(face)
        hsv[...,1] = 255
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flows.append(rgb)
        prev_gray = gray
    return flows

def predict_images(model, flows):
    preds = []
    for img in flows:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            preds.append(pred)
    return preds

# ==== STREAMLIT UI ====
st.title("👁️‍🗨️ DeepFake Detector: Optical Flow + Vision Transformer")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.info("🔍 Menjalankan deteksi wajah & ekstraksi Optical Flow...")
    faces = extract_faces(video_path)
    flows = compute_farneback(faces)

    st.subheader("📸 Contoh Crop Wajah")
    st.image([faces[i] for i in range(min(5, len(faces)))], width=150)

    st.subheader("🌪️ Contoh Optical Flow (Farneback)")
    st.image([flows[i] for i in range(min(5, len(flows)))], width=150)

    model_choice = st.selectbox("Pilih model klasifikasi:", ["mobilenetv3", "vit", "cvit"])
    if st.button("🚀 Deteksi Kelas"):
        model = load_model(model_choice)
        predictions = predict_images(model, flows)
        label_map = {0: "Real", 1: "Fake"}
        majority = max(set(predictions), key=predictions.count)

        st.success(f"Hasil Majority Vote: **{label_map[majority]}**")
        st.write("Detail Prediksi per Frame:")
        st.write([label_map[p] for p in predictions])
