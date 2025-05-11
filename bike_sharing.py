import pandas as pd  # veri analizi ve DataFrame yapısı için , verileri tablo şeklinde işler
import numpy as np  # sayısal işlemler için
import matplotlib.pyplot as plt  # grafik çizimi için
import seaborn as sns  # grafiklerin görsel kalitesini arttırmak için 

# makine öğrenmesi için gerekli sklearn bileşenleri
from sklearn.model_selection import train_test_split    # veriyi 'eğitim' ve 'test' olarak ayrımak için
from sklearn.preprocessing import MinMaxScaler    # normalizasyon için (verileri 0-1 arasına sıkıştırma)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score    # performans ölçümleri için 
# gerçek ve tahmin edilen değerleri karşılaştırma (confusion_matrix), doğru tahmin oranı (accuracy_score)

from sklearn.tree import DecisionTreeClassifier    # karar ağacı algoritması 
from sklearn.ensemble import RandomForestClassifier    # random forest algoritması 
from sklearn.neighbors import KNeighborsClassifier    # knn algoritması
from sklearn.svm import SVC    # destek vektör makineleri
from sklearn.linear_model import LogisticRegression    # lojistik regresyon
from sklearn.metrics import classification_report    # sınıflandırma raporları için (precision, recall, f1-score)


# -------------------- Veriyi Yükleme ve Ön Hazırlık --------------------------------------------------------------------------
gunluk_veri = pd.read_csv('day.csv')
saatlik_veri = pd.read_csv('hour.csv')

# NORMALİZASYON yapılacak sütunları tanımlıyoruz
# sıcaklık, hissedilen sıcaklık, nem ve rüzgar hızını ölçekleyerek modeller için uygun hale getiriyoruz
olcek_sutunlari = ['temp', 'atemp', 'hum', 'windspeed']

# MinMaxScaler ile sütunların normalizasyonu
scaler = MinMaxScaler() # değerler 0-1 arasına çekilir
saatlik_veri[olcek_sutunlari] = scaler.fit_transform(saatlik_veri[olcek_sutunlari]) # değerleri (0,1) aralığına getirme

# kiralama seviyelerini sınıflandırıyoruz
# 'cnt(toplam saatlik bisiklet kiralama sayısı)' sütununa göre 3 eşit sınıf : az->0, orta->1, çok->2 
saatlik_veri['kiralama_seviyesi'] = pd.qcut(saatlik_veri['cnt'], q=3, labels=[0,1,2]) # sınıflandırma işlemi

# Grafik açıklamalarının saklanacağı liste
grafik_raporlari = []


# -------------------- Grafik 1 : Saatlik Kayıtlı vs Kayıtsız ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(data=saatlik_veri, x='hr', y='registered', label='Kayıtlı')  # kayıtlı kullanıcılar 
sns.lineplot(data=saatlik_veri, x='hr', y='casual', label='Kayıtsız')     # kayıtsız kullanıcılar
# lineplot: çizgi grafiği
# başlık ve eksen isimleri:
plt.title("Saatlik Kiralama Sayısı (Kayıtlı vs Kayıtsız)")
plt.xlabel("Saat")
plt.ylabel("Kiralama Sayısı")
plt.legend() # çizgilerin anlamını belirlemek için sağ üste çizgi ekler
plt.grid() # arkaya ızgara ekler
plt.show() 
grafik_raporlari.append("Grafik 1: Kayıtlı kullanıcılar genellikle sabah ve akşam saatlerinde zirve yaparken, kayıtsız kullanıcılar öğleden sonraları daha aktif.")


# -------------------- Grafik 2 : Açık Hava (weathersit=1) - Sıcaklık vs Kiralama ------------------------------------------------
plt.figure(figsize=(8,6))
acik_hava = saatlik_veri[saatlik_veri['weathersit'] == 1]  # açık hava 
sns.scatterplot(data=acik_hava, x='temp', y='cnt', color='green') # scatterplot: nokta grafiği
plt.title("Açık Hava - Sıcaklık vs Kiralama Sayısı")
plt.xlabel("Sıcaklık (normalize)")
plt.ylabel("Kiralama Sayısı")
plt.grid()
plt.show()
grafik_raporlari.append("Grafik 2: Açık havalarda sıcaklık arttıkça kiralama sayısı genel olarak artmaktadır.")

# -------------------- Grafik 3 : Bulutlu/Sisli Hava (weathersit=2) - Nokta Grafiği ---------------------------------------------
plt.figure(figsize=(8,6)) # filtreleme işlemi
bulutlu_hava = saatlik_veri[saatlik_veri['weathersit'] == 2]  # bulutlu hava (filtrelenmiş veriler burda tutulur)
sns.scatterplot(data=bulutlu_hava, x='hum', y='cnt', color='orange')
plt.title("Bulutlu/Sisli Hava - Nem vs Kiralama Sayısı (Nokta Grafiği)")
plt.xlabel("Nem (normalize)")
plt.ylabel("Kiralama Sayısı")
plt.grid()
plt.show()
grafik_raporlari.append("Grafik 3: Bulutlu veya sisli havalarda nem oranı arttıkça kiralama sayısı hafif azalıyor gibi görünüyor.")


# -------------------- Grafik 4 : Yağmurlu Hava (weathersit=3) - Sıcaklık vs Kiralama --------------------------------------------
plt.figure(figsize=(8,6))
yagmurlu_hava = saatlik_veri[saatlik_veri['weathersit'] == 3]  # yağmurlu hava 
sns.scatterplot(data=yagmurlu_hava, x='temp', y='cnt', color='blue')
plt.title("Yağmurlu Hava - Sıcaklık vs Kiralama Sayısı")
plt.xlabel("Sıcaklık (normalize)")
plt.ylabel("Kiralama Sayısı")
plt.grid()
plt.show()
grafik_raporlari.append("Grafik 4: Yağmurlu havalarda sıcaklık yükselse bile kiralama sayısı genellikle düşüktür.")


# -------------------- Grafik 5 : Yoğun Yağışlı Hava (weathersit=4) varsa - Nokta Grafiği ------------------------------------------
if 4 in saatlik_veri['weathersit'].unique(): # 4: yoğun yağışlı
    yogun_yagis = saatlik_veri[saatlik_veri['weathersit'] == 4]
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=yogun_yagis, x='temp', y='cnt', marker='X', s=80, color='red') # s: piksel
    plt.title("Yoğun Yağışlı Hava - Sıcaklık vs Kiralama Sayısı (Profesyonel Gösterim)")
    plt.xlabel("Sıcaklık (normalize)")
    plt.ylabel("Kiralama Sayısı")
    plt.grid()
    plt.show()
    grafik_raporlari.append("Grafik 5: Yoğun yağışlı havalarda kiralama sayısı belirgin şekilde azalmaktadır.")
else:
    grafik_raporlari.append("Grafik 5: Yoğun yağışlı hava (weathersit=4) veride bulunamadı.")


# -------------------- Grafik 6 : Aylara Göre Kiralama Sayıları (2011 vs 2012) ------------------------------------------------------
plt.figure(figsize=(14, 5))

# 2011
plt.subplot(1, 2, 1) # yan yana 2 grafikten 1.yi yapıyoruz
veri_2011 = gunluk_veri[gunluk_veri['yr'] == 0] # yr=0 : 2011
aylik_2011 = veri_2011.groupby('mnth')['cnt'].sum() # aynı aya sahip satırları gruplar
sns.barplot(x=aylik_2011.index, y=aylik_2011.values, palette='Blues')
# barplot: çubuk grafiği  # x ekseni: aylar, y ekseni: kiralama sayıları
plt.title("2011 - Aylara Göre Kiralama Sayısı")
plt.xlabel("Ay")
plt.ylabel("Toplam Kiralama")

# 2012
plt.subplot(1, 2, 2) # 2.grafik
veri_2012 = gunluk_veri[gunluk_veri['yr'] == 1] # yr=1: 2012
aylik_2012 = veri_2012.groupby('mnth')['cnt'].sum()
sns.barplot(x=aylik_2012.index, y=aylik_2012.values, palette='Greens')
plt.title("2012 - Aylara Göre Kiralama Sayısı")
plt.xlabel("Ay")
plt.ylabel("Toplam Kiralama")

plt.tight_layout()  # iki grafiğin birbirine çakışmaması için
plt.show()
grafik_raporlari.append("Grafik 6: 2011 ve 2012 yıllarında aylara göre kiralama miktarları karşılaştırıldı.")


# -------------------- Grafik 7 : Haftanın Günlerine Göre Kiralama Sayıları (2011 vs 2012) ------------------------------------------
plt.figure(figsize=(14, 5))

gun_isimleri = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']

# 2011
plt.subplot(1, 2, 1)
haftalik_2011 = veri_2011.groupby('weekday')['cnt'].sum()
sns.barplot(x=gun_isimleri, y=haftalik_2011.values, palette='Purples')
plt.title("2011 - Günlere Göre Kiralama Sayısı")
plt.xticks(rotation=45)
plt.ylabel("Toplam Kiralama")

# 2012
plt.subplot(1, 2, 2)
haftalik_2012 = veri_2012.groupby('weekday')['cnt'].sum()
sns.barplot(x=gun_isimleri, y=haftalik_2012.values, palette='Oranges')
plt.title("2012 - Günlere Göre Kiralama Sayısı")
plt.xticks(rotation=45)
plt.ylabel("Toplam Kiralama")

plt.tight_layout()
plt.show()
grafik_raporlari.append("Grafik 7: 2011 ve 2012 yıllarında hafta günlerine göre kiralama dağılımı karşılaştırıldı.")



# -------------------- Grafik 8 : Boxplot ile Aykırı Değerler ------------------------------------------------------------------
plt.figure(figsize=(12, 8))

# Her sayısal sütun için bir boxplot
for i, sutun in enumerate(olcek_sutunlari):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=saatlik_veri[sutun], color='lightblue')
    plt.title(f"{sutun} - Boxplot")
    plt.tight_layout()

plt.show()
grafik_raporlari.append("Grafik 8: Boxplot grafiklerine göre özellikle nem ve rüzgar hızında aykırı değerler gözlenmektedir.")


from scipy.stats import zscore

# Z-score hesaplama
z_degerleri = saatlik_veri[olcek_sutunlari].apply(zscore)

# Z-score değeri 3’ten büyük olanlar potansiyel aykırı değer kabul edilir
aykiri_maskesi = (np.abs(z_degerleri) > 3)

# Aykırı değer sayısını yazdırma
print("\n--- Aykırı Değer Sayısı (Z-Score > 3) ---")
for sutun in olcek_sutunlari:
    aykiri_sayi = aykiri_maskesi[sutun].sum()
    print(f"{sutun}: {aykiri_sayi} aykırı değer")


# -------------------- Makine Öğrenmesi --------------------------------------------------------------------------------------------
degiskenler = ['season', 'hr', 'temp', 'hum', 'windspeed', 'weekday']  # girdi değişkenleri
# model öğrenmesi için istediğmiz değişkenler (eğiteceğimiz)
X = saatlik_veri[degiskenler]   # bağımsız değişkenler
y = saatlik_veri['kiralama_seviyesi']   # hedef sınıf (kiralama tahmin edilecek)

# veriyi eğitim ve test kümelerine ayırıyoruz (%70-eğitim , %30-test)
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# farklı makine öğrenme modelleri
#  kullanılan modelleri sözlük şeklinde tanımladık
modeller = {
    "Karar Ağacı": DecisionTreeClassifier(max_depth=5),
    "Rastgele Orman": RandomForestClassifier(n_estimators=100), # 100 tane karar ağacı
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF Çekirdek)": SVC(kernel='rbf'), #çekirdek kullanılır,
    # doğrusal olmayan sınırlara sahip verilerde sınıflandırma yapmak için
    "Lojistik Regresyon": LogisticRegression(max_iter=200)
}
# sonuçların saklanacağı sözlük 
sonuclar = {}

# -------------------- Model Eğitimi ve Değerlendirmesi ------------------------------------------------------------------------
for isim, model in modeller.items():
    model.fit(X_egitim, y_egitim)                     # modeli eğitme
    y_tahmin = model.predict(X_test)                  # tahmin etme
    dogruluk = accuracy_score(y_test, y_tahmin)       # doğruluk hesaplama
    sonuclar[isim] = dogruluk                         # sonuçları kaydetme

    # karmaşıklık matrislerini görselleştirme
    cm = confusion_matrix(y_test, y_tahmin)
    gosterim = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Az', 'Orta', 'Çok'])
    gosterim.plot(cmap='Blues')
    plt.title(f"{isim} - Karmaşıklık Matrisi (Doğruluk: {dogruluk:.2f})")
    plt.show()
    # bu matris ile modellerin hangi sınıflarda doğru ve yanlış tahminler yaptığını gösterir

# -------------------- Performans Karşılaştırma Grafiği ----------------------------------------------------------------------
# tüm modellerin doğruluklarını çubuk grafiği ile karşılaştırıyoruz
plt.figure(figsize=(10,6))
sns.barplot(x=list(sonuclar.keys()), y=list(sonuclar.values()), palette='viridis')
# x ekseni için: model isimleri, y ekseni için: skorlar
plt.title("Modellerin Doğruluk Karşılaştırması")
plt.ylabel("Doğruluk Skoru")
plt.ylim(0, 1) # y ekseninin sınırları
plt.xticks(rotation=15)
plt.grid(axis='y') # ızgarayı sadece y eksenine paralel ekler
plt.show()


# -------------------- Terminalde Grafik ve Model Raporları ------------------------------------------------------------------
print("\n--- Grafik Raporları ---\n")
for rapor in grafik_raporlari:
    print(rapor)
# doğruluk seviyesine göre yorumlama
print("\n--- Model Performans Karşılaştırmaları ---\n")
for model, dogruluk in sonuclar.items():
    yorum = ""
    if dogruluk >= 0.75:
        yorum = " Yüksek doğruluk (güvenli)"
    elif dogruluk >= 0.65:
        yorum = " Orta seviye doğruluk (bazı durumlar için yeterli)"
    else:
        yorum = " Düşük doğruluk (geliştirilmeli)"
    print(f"{model}: {dogruluk:.2f} - {yorum}")


# -------------------- Her Model İçin Detaylı Sonuç ve Sınıflandırma Raporu --------------------------------------------------
print("\n--- Her Model İçin Detaylı Sonuç ve Sınıflandırma Raporu ---")

for isim, model in modeller.items():
    print(f"\n Model: {isim}")
    
    y_tahmin = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_tahmin)

    yorum = ""
    if dogruluk >= 0.75:
        yorum = " Yüksek doğruluk"
    elif dogruluk >= 0.65:
        yorum = " Orta seviye doğruluk"
    else:
        yorum = " Düşük doğruluk"
# tahmin sonuçlarını ve gerçek değerlerle uyuşmazlıklarını tablo olarak verir
    cm = confusion_matrix(y_test, y_tahmin)
    cm_df = pd.DataFrame(cm, index=["Gerçek Az", "Gerçek Orta", "Gerçek Çok"],
                             columns=["Tahmin Az", "Tahmin Orta", "Tahmin Çok"])
# pd.DataFrame daha okunabilir olması için matrisi tabloya dönüştürür
    print("\n Doğruluk Skoru:")
    print(f"{dogruluk:.2f}") # doğruluk skoru 2 basamaklı ondalık sayıyla yazılır.
    
    print("\n Karışıklık Matrisi:")
    print(cm_df.to_string()) # karmaşıklık matrisi tablo formatında ekrana yazdırılır

    print("\n Sınıflandırma Raporu:")
    print(classification_report(y_test, y_tahmin, target_names=["Az", "Orta", "Çok"]))
# classification_report sayesinde hangi sınıfta ne kadar başarılı olmuş öğrenilir
    print("\n Açıklama:")
    print(yorum)

#----------------------------Yapay Sinir Ağı Modeli (MLPRegressor)--------------------------------
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Girdi ve hedef değişken
X = saatlik_veri.drop(columns=["cnt", "casual", "registered", "dteday"], errors='ignore')
X = X.select_dtypes(include=[np.number])  # sayısal sütunlar
y = saatlik_veri["cnt"]

# 2. Normalize et ve tekrar DataFrame'e çevir
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3. Eğitim ve test verisine ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Yapay sinir ağı modeli tanımla
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32),
                         activation='relu',
                         solver='adam',
                         learning_rate_init=0.001,
                         max_iter=500,
                         batch_size=32,
                         random_state=42,
                         verbose=False)

# 5. Eğit
mlp_model.fit(X_train, y_train)

# 6. Tahmin ve performans
y_pred = mlp_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Yapay Sinir Ağı Performansı ---")
print(f"Ortalama Kare Hata (MSE): {mse:.2f}")
print(f"R2 Skoru: {r2:.2f}")

# 7. Öğrenme eğrisi
loss_curve = mlp_model.loss_curve_
plt.figure(figsize=(10, 5))
plt.plot(loss_curve, color='blue', label='Eğitim Kaybı')
plt.xlabel("Epoch")
plt.ylabel("Kayıp (Loss)")
plt.title("Yapay Sinir Ağı Öğrenme Eğrisi")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 8. Tahmin vs Gerçek Değer Grafiği
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Gerçek", color="green", marker='o')
plt.plot(y_pred[:100], label="Tahmin", color="red", linestyle='dashed', marker='x')
plt.title("Yapay Sinir Ağı: Gerçek vs Tahmin Edilen Kiralama Sayısı (İlk 100 Gözlem)")
plt.xlabel("Gözlem Sırası")
plt.ylabel("Kiralama Sayısı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#--------------------------RandomForestRegressor ile Değişken Önemi --------------------------------------
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

# Yeni X: sadece anlamlı değişkenleri içeren
# Örneğin: saat, tatil günü, çalışma günü, hava durumu, sıcaklık, hissedilen sıcaklık, nem, rüzgar hızı
X_filtered = X_train[['hr', 'holiday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
rf = RandomForestRegressor(random_state=42)
rf.fit(X_filtered, y_train)

# Değişken önem dereceleri
importances = rf.feature_importances_
feature_names = X_filtered.columns

# Türkçeleştirme sözlüğü
degisken_adlari = {
    'hr': 'Saat',
    'holiday': 'Tatil Günü',
    'workingday': 'Çalışma Günü',
    'weathersit': 'Hava Durumu',
    'temp': 'Sıcaklık',
    'atemp': 'Hissedilen Sıcaklık',
    'hum': 'Nem',
    'windspeed': 'Rüzgar Hızı'
}

translated_names = [degisken_adlari.get(name, name) for name in feature_names]
importance_df = pd.DataFrame({"Değişken": translated_names, "Önem Skoru": importances})
importance_df = importance_df.sort_values("Önem Skoru", ascending=False)

# Tablo ve grafik
print("Bisiklet Kiralama Sayısını Etkileyen Değişkenler (Önem Skoru)")
print(importance_df)

plt.figure(figsize=(10,6))
plt.barh(importance_df["Değişken"], importance_df["Önem Skoru"], color='skyblue')
plt.xlabel("Önem Skoru")
plt.title("Değişkenlerin Bisiklet Kiralama Sayısına Etkisi")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


