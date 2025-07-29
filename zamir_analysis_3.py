import re

TURKCE_ZAMIRLER = [
    "ben", "sen", "o", "biz", "siz", "onlar",
    "bu", "şu", "bunu", "şunu", "onu",
    "kendim", "kendin", "kendisi", "kendimiz", "kendiniz", "kendileri",
    "kendi", "hepimiz", "hepiniz", "hepsi", "herkes", "hiçkimse"
]

# ---------- Yardımcı fonksiyonlar ---------- #
def zamir_orani_hesapla(metin):
    kelimeler = re.findall(r"\b\w+\b", metin.lower())
    toplam = len(kelimeler)
    zamir_say = sum(1 for k in kelimeler if k in TURKCE_ZAMIRLER)
    return zamir_say / toplam if toplam else 0

def zamir_belirsizligi_var_mi(metin):
    for cumle in re.split(r"[.!?]", metin):
        kelimeler = re.findall(r"\b\w+\b", cumle.lower())
        if kelimeler.count("o") > 1 or kelimeler.count("kendi") > 1:
            return True
    return False

UYUM_HARITASI = {
    "ben":   ["um", "ım", "im", "üm", "dum", "dim", "düm", "dum"],
    "sen":   ["sun", "sün"],
    "o":     ["",],  # tekil, özel ek beklenmez
    "biz":   ["uz", "ız", "iz", "üz"],
    "siz":   ["sunuz", "sünüz"],
    "onlar": ["lar", "ler"]
}

def zamir_uyumsuzlugu_var_mi(metin):
    for cumle in re.split(r"[.!?]", metin):
        words = re.findall(r"\b\w+\b", cumle.lower())
        for zamir, ekler in UYUM_HARITASI.items():
            if zamir in words:
                # son kelime(=yüklem) ek uygun mu?
                yuklem = words[-1]
                if zamir == "o":  # tekil 'o' için çoğul ek kontrolü
                    if yuklem.endswith(("lar", "ler")):
                        return True
                else:
                    if not any(yuklem.endswith(e) for e in ekler):
                        return True
    return False

# ---------- Ana fonksiyon ---------- #
def alzheimer_risk_degerlendir(metin):
    oran = zamir_orani_hesapla(metin)
    belirsiz = zamir_belirsizligi_var_mi(metin)
    uyumsuz  = zamir_uyumsuzlugu_var_mi(metin)

    # Basit skorlama
    if oran < 0.05 and not (belirsiz or uyumsuz):
        risk = 1
    elif oran < 0.10 and not uyumsuz:
        risk = 2
    else:
        risk = 3

    return {
        "zamir_orani": round(oran, 3),
        "belirsizlik": belirsiz,
        "uyumsuzluk":  uyumsuz,
        "risk_skoru":  risk
    }

# ---------- Test ---------- #
if __name__ == "__main__":
    metin = input("Metni girin: ")
    sonuc = alzheimer_risk_degerlendir(metin)

    print("\n--- Analiz ---")
    print("Zamir oranı:", sonuc["zamir_orani"])
    print("Belirsizlik:", "Var" if sonuc["belirsizlik"] else "Yok")
    print("Uyumsuzluk:", "Var" if sonuc["uyumsuzluk"] else "Yok")
    print("Risk skoru:", sonuc["risk_skoru"])
