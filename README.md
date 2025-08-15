

# 🧠 Alzheimer Risk Tahmini ve Bilişsel Dil Analizi Uygulaması

Bu uygulama, Doğal Dil İşleme (NLP) ve makine öğrenmesi modelleri kullanarak, kullanıcıdan alınan metinler üzerinde bilişsel dil analizi gerçekleştirir. Amaç, Alzheimer gibi bilişsel bozuklukların erken belirtileri olabilecek dilsel sapmaları tespit etmeye yardımcı olmak ve bir ön risk değerlendirmesi sunmaktır.

-----

## 🚀 Özellikler

  * *Çok Modelli Analiz:* Birden fazla NLP modelinin (BERT tabanlı zamir.modeli, sond modeli) tahminlerini birleştirerek daha kapsamlı bir değerlendirme sunar.
  * *Konsolide Risk Çıktısı:* Farklı analiz yöntemlerinden elde edilen sonuçları tek bir "Düşük Risk", "Orta Risk" veya "Yüksek Risk" seviyesinde birleştirir.
  * *Görsel Destekli Soru Seti:* Kullanıcıya, anılarını ve düşüncelerini daha ayrıntılı anlatmasını teşvik etmek için hem açık uçlu sorular hem de görsel sorular sunar.
  * *Kullanıcı Dostu Arayüz:* Streamlit kütüphanesi ile oluşturulmuş basit ve etkileşimli bir web arayüzüne sahiptir.

-----

## 🛠️ Gereksinimler

Uygulamayı yerel bilgisayarınızda çalıştırmak için aşağıdaki yazılımların yüklü olması gerekmektedir:

  * *Python 3.8+*
  * *Git*

-----

## 📦 Kurulum ve Çalıştırma

Projenin tüm bağımlılıklarını ve modellerini doğru bir şekilde kurmak için aşağıdaki adımları sırasıyla uygulayın.

### 1\. Depoyu Klonlama

Projenin kodunu bilgisayarınıza indirmek için terminalde aşağıdaki komutu çalıştırın:

bash
git clone [DEPO_ADRESİ]
cd [DEPO_ADRESİ]


> *Not:* [DEPO_ADRESİ] yerine arkadaşınızın projesinin GitHub URL'sini yapıştırın.

### 2\. Gerekli Dosyaları İndirme

Uygulamanın çalışması için gerekli olan model dosyalarını ve görselleri indirip doğru klasöre yerleştirmeniz gerekmektedir.

> *Önemli:* Proje klasörünüzün içine random.modeller adında yeni bir klasör oluşturun. Tüm .zip model dosyalarını ve .jpeg görsel dosyalarını bu klasörün içine yerleştirin.

İndirmeniz gereken dosyalar şunlardır:

  * anlam.belirsizligi.zip
  * bert_alzheimer_tokenizer.zip
  * berturk_finetunedsst_model.zip
  * sond.zip
  * Untitled7.zip
  * zamir.modeli.zip
  * atom.logo.jpeg
  * kurabiye.hirsizi.jpeg

### 3\. Sanal Ortam Oluşturma (Önerilir)

Proje bağımlılıklarını sistem genelindeki paketlerden izole etmek için bir sanal ortam oluşturun:

bash
python -m venv venv


Sanal ortamı etkinleştirin:

  * *Windows:* venv\Scripts\activate
  * *macOS/Linux:* source venv/bin/activate

### 4\. Bağımlılıkları Kurma

Aşağıdaki komutla Streamlit ve diğer tüm gerekli Python kütüphanelerini kurun. Bunun için projenin kök dizininde requirements.txt adında bir dosya oluşturup aşağıdaki içeriği içine kopyalayın:

txt
streamlit
plotly
transformers
torch
accelerate
pandas
numpy
joblib


Ardından, sanal ortamınız aktifken terminalde aşağıdaki komutu çalıştırın:

bash
pip install -r requirements.txt


### 5\. Uygulamayı Çalıştırma

Tüm kurulumlar tamamlandıktan sonra, projenin ana klasöründe terminali açın ve uygulamayı başlatmak için şu komutu kullanın:

bash
streamlit run [ANA_PYTHON_DOSYASI_ADİ.py]


> *Not:* [ANA_PYTHON_DOSYASI_ADİ.py] yerine sizin Streamlit kodunuzun bulunduğu dosyanın adını yazın (örneğin main.py veya arayuz.py).

Tarayıcınızda uygulamanız otomatik olarak açılacaktır.

-----

## ⚠️ Önemli Not

Gönderdiğiniz kod, model dosyalarını otomatik olarak C:\random.modeller\zamir.modeli klasörüne çıkartmaktadır. Çalıştırma sırasında bu klasörde bert_alzheimer_tokenizer, zamir.modeli ve sond modelleri kullanılmaktadır. Diğer ZIP dosyaları (anlam.belirsizligi, Untitled7 ve berturk_finetunedsst_model) çıkartılsa da, kodunuz bu modelleri analiz için kullanmamaktadır. Eğer bu modelleri de kullanmak isterseniz, load_all_models ve konsolide_analiz_yap fonksiyonlarını bu modellere uygun şekilde güncellemeniz gerekecektir.
