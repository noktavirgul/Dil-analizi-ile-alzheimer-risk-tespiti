import streamlit as st
import plotly.express as px
import os
import zipfile
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
import torch
from accelerate import Accelerator
from transformers import BitsAndBytesConfig
import base64
from collections import Counter
import re
import pandas as pd
import joblib 
import numpy as np 
from pathlib import Path

# ZIP dosyası yolları ve çıkartılacak klasör
# Lütfen tüm ZIP ve görsel dosyalarının bu klasörde olduğundan emin olun.
model_zip_paths = {
    "bert_tokenizer_zip_path": "C:\\random.modeller\\bert_alzheimer_tokenizer.zip",
    "zamir_modeli_zip_path": "C:\\random.modeller\\zamir.modeli.zip",
    "sond_zip_path": "C:\\random.modeller\\sond.zip",
    "bert_alzheimer_tokanizer_zip_path": "C:\\random.modeller\\bert_alzheimer_tokenizer.zip",
    "untitled7_zip_path": "C:\\random.modeller\\Untitled7.zip"
}
zamir_extract_dir = "C:\\random.modeller\\zamir.modeli"

# sond.py modeli için gerekli dosyalar
id2label = {0: "Normal", 1: "Alzheimer"}

# ZIP dosyasını çıkartma işlemini bir kez yapalım
def extract_zip(zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
    
    try:
        if not zipfile.is_zipfile(zip_path):
            st.error(f"Dosya bir ZIP dosyası değil: {zip_path}. Lütfen dosya türünü kontrol edin.")
            st.stop()

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        # st.success mesajı kaldırıldı
    except FileNotFoundError:
        st.error(f"Model ZIP dosyası bulunamadı: {zip_path}. Lütfen dosya yolunu kontrol edin.")
        st.stop()
    except zipfile.BadZipFile:
        st.error(f"Geçersiz ZIP dosyası: {zip_path}. Dosyanın bozuk olmadığından emin olun.")
        st.stop()
    except Exception as e:
        st.error(f"Dosya çıkartılırken bir hata oluştu ({os.path.basename(zip_path)}): {e}")
        st.stop()

# Tüm tanımlı ZIP dosyalarını çıkart
# st.info mesajı kaldırıldı
for path in model_zip_paths.values():
    extract_zip(path, zamir_extract_dir)


# ----------------------------------------------------
# MODELLERİN YÜKLENMESİ VE ÖNBELLEKLEME
# ----------------------------------------------------

@st.cache_resource
def load_all_models():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    zamir_model, zamir_tokenizer, sond_model, sond_tokenizer = None, None, None, None

    try:
        zamir_model = BertForSequenceClassification.from_pretrained(
            zamir_extract_dir,
            quantization_config=quantization_config,
            device_map="auto"
        )
        zamir_tokenizer = BertTokenizer.from_pretrained(zamir_extract_dir)
        
        sond_model = AutoModelForSequenceClassification.from_pretrained(zamir_extract_dir, num_labels=2)
        sond_tokenizer = AutoTokenizer.from_pretrained(zamir_extract_dir)

        zamir_model.eval()
        sond_model.eval()
        # st.success mesajı kaldırıldı
    except Exception as e:
        st.error(f"Modeller yüklenirken bir hata oluştu: {e}. Dosya adlarının ve klasör içeriğinin doğru olduğundan emin olun.")
        st.stop()
        
    return zamir_model, zamir_tokenizer, sond_model, sond_tokenizer

# Modelleri ve tokenizer'ları bir kez yükle
(zamir_model, zamir_tokenizer, 
 sond_model, sond_tokenizer) = load_all_models()

# Cihaz belirleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# ANALİZ FONKSİYONLARI
# ----------------------------------------------------

def sond_analizi_yap(metin, model, tokenizer):
    if not model or not tokenizer:
        return []
    
    metin_temiz = re.sub(r"[.,!?;:\"“”’'()]", " ", metin.lower())
    kelimeler = [w for w in metin_temiz.split() if w.strip()]
    sayac = Counter(kelimeler)
    toplam = len(kelimeler)
    
    if toplam == 0:
        return []

    sonuc_df = pd.DataFrame(columns=["Kelime", "Tahmin"])
    
    for kelime, adet in sayac.items():
        giris = metin.replace(kelime, f"[VURGULANMIŞ_KELİME] {kelime} [VURGULANMIŞ_KELİME]")
        enc = tokenizer(giris, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
        logits = model(**enc).logits
        tahmin_etiketi = int(logits.argmax(dim=1).item())
        tahmin_metni = id2label[tahmin_etiketi]
        sonuc_df.loc[len(sonuc_df)] = [kelime, tahmin_metni]

    return sonuc_df

def konsolide_analiz_yap(metin, zamir_model, zamir_tokenizer, sond_model, sond_tokenizer):
    risk_seviyesi = 0 # 0: Düşük, 1: Orta, 2: Yüksek

    # 1. Zamir Modeli Tahmini
    if zamir_model and zamir_tokenizer:
        inputs = zamir_tokenizer(metin, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = zamir_model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        
        if prediction == 1: # Orta Risk
            risk_seviyesi = max(risk_seviyesi, 1)
        elif prediction == 2: # Yüksek Risk
            risk_seviyesi = max(risk_seviyesi, 2)
            
    # 2. Sond.py Modeli Analizi
    if sond_model and sond_tokenizer:
        sond_sonuc_df = sond_analizi_yap(metin, sond_model, sond_tokenizer)
        if not sond_sonuc_df.empty:
            alzheimer_kelime_sayisi = (sond_sonuc_df['Tahmin'] == 'Alzheimer').sum()
            toplam_kelime = len(sond_sonuc_df)
            if toplam_kelime > 0 and (alzheimer_kelime_sayisi / toplam_kelime) > 0.1: # %10'dan fazlaysa
                risk_seviyesi = max(risk_seviyesi, 1)
            if toplam_kelime > 0 and (alzheimer_kelime_sayisi / toplam_kelime) > 0.25: # %25'ten fazlaysa
                risk_seviyesi = max(risk_seviyesi, 2)

    # 3. Manuel Zamir Analizi
    zamirler = ["ben", "sen", "o", "biz", "siz", "onlar"]
    zamir_sonuc = {zamir: metin.lower().split().count(zamir) for zamir in zamirler}
    toplam_zamir = sum(zamir_sonuc.values())
    if toplam_zamir > 0:
        ben_oran = zamir_sonuc.get("ben", 0) / toplam_zamir
        o_oran = zamir_sonuc.get("o", 0) / toplam_zamir
        if ben_oran > 0.4 or o_oran > 0.3:
            risk_seviyesi = max(risk_seviyesi, 1)

    return risk_seviyesi

# ----------------------------------------------------
# STREAMLIT ARAYÜZ KISMI
# ----------------------------------------------------

st.set_page_config(page_title="Alzheimer Tahmini", layout="wide")

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

logo_path = r"C:\veri.modelleri\atom.logo.jpeg"
logo_base64 = get_base64_image(logo_path)

if logo_base64:
    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: center; padding: 10px 0;">
            <img src="data:image/jpeg;base64,{logo_base64}" style="width:200px; border-radius:10px;">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.error("Logo dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")

st.sidebar.title("Menü")

sayfa = st.sidebar.radio(
    label="",
    options=["🏠 Ana Sayfa", "🧠 Analiz Yap", "❓ Nasıl Çalışır", "📞 İletişim"],
    label_visibility="hidden"
)

if sayfa == "🏠 Ana Sayfa":
    st.title("🧠 Alzheimer Risk Tahmini")
    st.subheader("Doğal Dil İşleme Tabanlı Bilişsel Dil Analizi")
    st.markdown("""
        Bu uygulama, kullanıcıdan alınan metinleri zamir kullanımı, cümle uzunluğu, kelime tekrarı, bağlaç kullanımı ve anlamsal tutarlılık gibi dilsel ölçütlere göre analiz eder.
        Doğal Dil İşleme (NLP) ve makine öğrenmesi teknikleri kullanılarak, metindeki bilişsel bozukluk belirtilerini tespit etmeye yardımcı olur.
        Amacımız, kullanıcıya erken uyarı niteliğinde bir risk değerlendirmesi sunmak ve gerektiğinde sağlık uzmanına yönlendirmektir. Bu sistem tıbbi tanı koymaz, yalnızca ön değerlendirme desteği sağlar.
    """)

elif sayfa == "🧠 Analiz Yap":
    st.header("📝 Açık Uçlu Sorularla Alzheimer Riski Analizi")
    st.markdown("Aşağıdaki soruları sırayla yanıtlayınız. Lütfen olabildiğince doğal ve ayrıntılı cevaplar verin.")

    sorular = [
        "1. Nasılsınız, gününüz nasıl geçti?",
        "2. Geçen hafta seni en çok mutlu eden şey neydi?",
        "3. Çocukluğunuzda en çok hatırladığınız anı nedir?",
        "4. Hayatınız boyunca sizi en çok etkileyen kişi kimdi? Neden?",
        "5. Ailenizle ilişkileriniz nasıl? Onlardan bahsedebilir misiniz?",
        "6. Eğer geçmişe dönüp bir şeyi değiştirebilseydiniz neyi değiştirirdiniz?",
        "7. Gelecek hafta için planların neler?",
        "8. Haftasonu hava çok güzel olursa neler yaparsın, kimlerle vakit geçirirsin?",
        "9. En büyük hayaliniz nedir?",
        "10. Bu görselde neler görüyorsunuz?"
    ]

    if 'soru_index' not in st.session_state:
        st.session_state.soru_index = 0
    if 'cevaplar' not in st.session_state:
        st.session_state.cevaplar = [""] * len(sorular)

    index = st.session_state.soru_index
    st.write(f"**{sorular[index]}**")

    if index == 9: 
        image_path = "C:\\veri.modelleri\\kurabiye.hirsizi.jpeg"
        if os.path.exists(image_path):
            st.image(image_path, caption="Soru 10 Görseli", width=450)
        else:
            st.error(f"Görsel bulunamadı: {image_path}")

    st.session_state.cevaplar[index] = st.text_area(
        "Cevabınız:", 
        value=st.session_state.cevaplar[index], 
        height=150,
        key=f"cevap_{index}"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if index > 0 and st.button("⬅️ Geri"):
            st.session_state.soru_index -= 1
            st.rerun()
            
    with col2:
        if index < len(sorular) - 1:
            if st.button("➡️ İleri"):
                st.session_state.soru_index += 1
                st.rerun()
        else:
            if st.button("🔍 Analizi Başlat"):
                metin = " ".join(st.session_state.cevaplar)

                if metin.strip() == "":
                    st.warning("Lütfen en az bir soruyu yanıtlayın.")
                else:
                    st.success("Analiz tamamlandı.")

                    st.markdown("---")
                    st.subheader("🧠 Konsolide Alzheimer Riski Tahmin Sonucu")

                    if zamir_model and zamir_tokenizer and sond_model and sond_tokenizer:
                        final_risk_seviyesi = konsolide_analiz_yap(metin, zamir_model, zamir_tokenizer, sond_model, sond_tokenizer)
                        
                        etiketler = {0: "🟢 Risk Yok", 1: "⚠️ Orta Risk", 2: "🔴 Yüksek Risk"}
                        
                        st.markdown(f"### **Tahmin: {etiketler.get(final_risk_seviyesi, 'Bilinmiyor')}**")
                        
                        st.markdown("""
                        **Not:** Bu sonuç, birden fazla modelin ve manuel analizin bulgularını birleştirerek oluşturulmuştur.
                        Bu bir tıbbi teşhis değildir ve yalnızca bilgilendirme amaçlıdır.
                        """)
                    else:
                        st.warning("Gerekli modeller yüklenemediği için analiz yapılamadı.")

elif sayfa == "❓ Nasıl Çalışır":
    st.header("🔧 Uygulama Nasıl Çalışıyor?")

    with st.expander("📌 Model Hakkında"):
        st.markdown("""
        - Alzheimer hastalığı yedi aşamada incelenir. Hastalık ilerledikçe dikkat eksikliği, anlama güçlüğü ve özellikle dil bozuklukları gibi semptomlar giderek belirginleşir.
        - Araştırmalar, Alzheimer tanısı konulmadan çok önce, bilişsel gerileme başlamadan önce bile temel dil işlevlerinde seçici bozulmaların ortaya çıktığını göstermektedir. Bu da dil bozukluklarının hastalığın erken ve özgün belirtilerinden biri olduğunu ortaya koymaktadır.
        - Konuşma transkriptleri üzerinde makine öğrenimi ve doğal dil işleme teknikleri kullanılarak Alzheimer riski değerlendirmesi yapılmaktadır. Bu yöntemler, hastalığın erken teşhisinde etkili ve giderek yaygınlaşan araçlar olarak önem kazanmaktadır.
        """)

    with st.expander("📖 Kaynakça"):
        st.markdown("""
        - https://onlinelibrary.wiley.com/doi/abs/10.1002/gps.3766?casa_token=AOxnPF4HTdoAAAAA:iVVSenAAV3yJmpm36O_9dzHdUmHfvko9-Z8CgBfXpBtN5U8FFb_ChcObapeiu9TpJF-LWAAHQJvVt8Y
        - E. Akarsu Ve Ark. , "Alzheimer's Disease Detection and Dataset Creation from Spontaneous Speech Spontane Konuşmadan Alzheimer Hastalığı Tespiti ve Veri Seti Oluşturma," 32nd IEEE Conference on Signal Processing and Communications Applications, SIU 2024 , Mersin, Türkiye, 2024
        """)

elif sayfa == "📞 İletişim":
    st.header("📧 Bize Ulaşın")
    st.markdown("Sorularınız veya iş birliği için bizimle iletişime geçebilirsiniz.")
    st.markdown("**E-posta:** 📫 atomnoktavirgul@gmail.com")
