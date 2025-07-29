import streamlit as st
from PIL import Image

# Arka plan rengini açık mavi yap
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f7ff;
    }
    /* Başlık renk */
    h1 {
        color: black;
    }
    /* Textarea label rengini siyah yap */
    label[for="text_area"] > div {
        color: black !important;
        font-weight: bold;
    }
    /* Textarea iç yazı rengi (placeholder ve yazılanlar) siyah */
    textarea {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo yükle (dosya yolunu kendine göre düzenle)
logo = Image.open("c:\\Users\\AKMAN AİLESİ\\Downloads\\WhatsApp Image 2025-07-26 at 12.35.41.jpeg")
st.image(logo, width=200)

# Başlık
st.markdown("<h1>🧠 Alzheimer Riski Tahmini (Zamir Analizi)</h1>", unsafe_allow_html=True)

# Basit zamir listesi
PRONOUNS = {
   "ben", "sen", "o", "biz", "siz", "onlar",
    "bana", "sana", "ona", "bizi", "sizi", "onları",
    "beni", "seni", "onu", "bize", "size", "onlara",
     "kendi", "kendim", "kendisi", "kendin", "kendimiz",
    "kendiniz", "kendileri",
    "kim", "ne", "hangisi", "kaçı", "kaçıncı",
    "bu", "şu", "bunlar", "şunlar",
    "biri", "birisi", "kimse", "hiç kimse", "herkes",
     "hepsi", "bazısı", "birkaç", "birkaçı", "tümü",
      "bazıları", "hiçbiri", "başkası"
}

def pronoun_ratio(text):
    words = text.lower().split()
    total = len(words)
    if total == 0:
        return 0
    pronouns = sum(1 for word in words if word in PRONOUNS)
    return pronouns / total

# Text area
user_input = st.text_area("Metin giriniz:", height=150, key="text_area")

# Buton ve analiz
if st.button("Analiz Et"):
    ratio = pronoun_ratio(user_input)
    st.write(f"🔍 Zamir oranı: `{ratio:.2f}`")
    if ratio >= 0.15:
        st.error("⚠ Risk Seviyesi: YÜKSEK")
    else:
        st.success("🟢 Risk Seviyesi: DÜŞÜK")