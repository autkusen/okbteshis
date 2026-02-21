import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import requests
import os

# --- MODEL İNDİRME AYARLARI ---
# GitHub Releases kısmındaki .pth dosyanızın "Direct Download" linkini buraya yapıştırın
# Örnek: https://github.com/kullanici/proje/releases/download/v1.0/okb_modeli.pth
MODEL_URL = "https://github.com/autkusen/okbteshis/releases/download/okb/okb_teshis_modeli.pth"
MODEL_PATH = "model_cache.pth"

@st.cache_resource
def download_and_load_model():
    # 1. Eğer model yerelde yoksa GitHub'dan indir
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model dosyası ilk kez indiriliyor, lütfen bekleyin..."):
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Model indirilemedi! Linki kontrol edin.")
                return None

    # 2. Mimariyi oluştur (Eğittiğiniz mimari: EfficientNet-B0)
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    # 3. Ağırlıkları yükle
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Modeli çağır
model = download_and_load_model()
