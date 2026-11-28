# Termal İHA Görüntülerinde Küçük Hedeflerin Derin Öğrenme ile Tespiti ve Doğal Dilde Raporlanması

Bu proje; İnsansız Hava Araçlarından (İHA) alınan termal görüntülerde küçük hedeflerin tespit edilmesini, sınıflandırılmasını ve elde edilen sonuçların operasyonel durum raporu olarak doğal dilde sunulmasını amaçlamaktadır. Çalışmada CNN tabanlı **YOLOv8n** ile Transformer tabanlı **RT-DETR-L** mimarileri, sızıntısız (leakage-free) bağımsız bir test kümesinde karşılaştırılmıştır.

**Kapsam:** Bitirme Projesi 1 kapsamında sızıntısız ve deterministik veri bölmesinin kurulması, YOLOv8n ve RT-DETR-L temel mimarilerinin aynı test kümesi üzerinde karşılaştırılması ve kural tabanlı doğal dil raporlama modülünün prototipinin geliştirilmesi tamamlanmıştır. Yüksek çözünürlüklü dilimleme (SAHI) ile küçük nesne başarımının iyileştirilmesi ve raporlama modülüne mekânsal konum betimlemelerinin eklenmesi gibi geliştirmeler Bitirme Projesi 2 iş paketine bırakılmıştır.

---

## Sistem Mimarisi ve İş Akışı

<pre>
┌───────────────────────────────────────────────────────────────────────┐
│        Birleştirilmiş İHA Veri Seti (27.925 -> 26.086 Görüntü)        │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Veri Sızıntısı Kontrolü ve Grup Bölme (MD5 / D4 dHash / NCC)         │
│  Bölme Parmak İzi: da43dd521edebb0d                                   │
└───────────────────┬───────────────────────────────────┬───────────────┘
                    │                                   │
                    ▼                                   ▼
      ┌───────────────────────────┐       ┌───────────────────────────┐
      │   Eğitim & Doğrulama      │       │   Bağımsız Test Kümesi    │
      │ (Train: 20.092 | Val: 3.020)│     │   (Test: 2974 Görüntü)    │
      └─────────────┬─────────────┘       └─────────────┬─────────────┘
                    │                                   │
          ┌─────────┴─────────┐                         │
          │                   │                         │
          ▼                   ▼                         │
  ┌───────────────┐   ┌───────────────┐                 │
  │ 1. Mimari:    │   │ 2. Mimari:    │                 │
  │ YOLOv8n (CNN) │   │ RT-DETR-L     │                 │
  └───────┬───────┘   └───────┬───────┘                 │
          │                   │                         │
          └─────────┬─────────┘                         │
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Test Değerlendirmesi ve Kıyaslama (mAP@50, mAP@50-95, AP_small)      │
└───────────────────┬───────────────────────────────────┬───────────────┘
                    │                                   │
                    ▼                                   ▼
      ┌───────────────────────────┐       ┌───────────────────────────┐
      │ Performans Metrikleri     │       │ Tespit Edilen Sınıflar    │
      │ (metrics_test.csv)        │       │ ['person', 'car', ...]    │
      └───────────────────────────┘       └─────────────┬─────────────┘
                                                        │
                                                        ▼
                                          ┌───────────────────────────┐
                                          │ Doğal Dil Modülü          │
                                          │ (report_generator.py)     │
                                          └─────────────┬─────────────┘
                                                        │
                                                        ▼
                                          ┌───────────────────────────┐
                                          │ Türkçe Durum Raporu       │
                                          │ (20 Örnek Çıktı)          │
                                          └───────────────────────────┘
</pre>

---

## 1. Veri Seti ve Kümeler

Başlangıçtaki birleştirilmiş açık kaynak veri setinde (`umuttuygurr/aerial-uav-thermal-inferred-unified-dataset`) yer alan **27.925** görüntü içerisinden, `gun` (ID: 6) sınıfına ait **1.839** görüntü çıkarılmıştır (`excluded_gun`). Bu görüntülerin tamamı `06_gundataset` kaynağındandır (orijinal bölmedeki dağılımları: `train` 1.471, `val` 181, `test` 187; kaynağın `gun` etiketi taşımayan kalan 61 görüntüsü veri setinde tutulmuştur). Çıkarma gerekçesi yazarın gözlemidir: `gun` sınıfı orijinal test bölmesinde yalnızca 187 görselle (%6.68) temsil edildiğinden sınıf bazlı AP'nin kararsız kaldığı ve genel `mAP@50` ortalamasını düşürdüğü görülmüştür; `gun` sınıfını içeren sızıntısız eğitim tamamlanmadan durdurulmuş, sızıntısız bölmede tam bir eğitim ve test ölçümü yapılmamıştır. Kalan **26.086** aktif görüntü (`train`: 20.092, `val`: 3.020, `test`: 2.974) ve **86.770** etiket, 6 hedef sınıf üzerinden analiz edilmiştir.

#### `gun` Sınıfının Çıkarılma Gerekçesi (Ayrıntı)

Yukarıdaki gerekçeye ek olarak raporda iki gözlem daha yer almaktadır:

- **Tek kaynak ve alan uyumsuzluğu:** `06_gundataset` görüntüleri İHA hava perspektifinden değil, silah odaklı farklı sahne ve çekim koşullarından oluşmaktadır; bu nedenle İHA tabanlı diğer hedef sınıfların operasyonel bağlamıyla örtüşmediği değerlendirilmiştir.
- **Ardışık kareler:** `gun` görüntüleri videolardan alınmış ardışık karelerden oluşmaktadır; yakın-kopya gruplamasında `1.839` görüntü yalnızca `21` gruba düşmektedir (en büyük üç grup `481`, `480` ve `297` görüntü).

Bu gözlemler ve örneklem azlığı göz önünde bulundurularak, ek eğitim ve hesaplama zamanı kaybı yaşanmaması adına `gun` sınıfını içeren sızıntısız bölmeyle başlatılan eğitim tamamlanmadan durdurulmuş ve sınıf veri setinden çıkarılmıştır.


Aşağıdaki tabloda, kümeler arası veri sızıntısını (data leakage) önlemek amacıyla yakın-kopya grupları bütünlüğü korunarak katmanlı (`stratified`) şekilde yeniden ayrılan birleştirilmiş İHA veri setinin (`merged_uav_dataset`) test kümesine ait görüntü ve etiket dağılımları yer almaktadır:

| Küme (Split) | Görüntü Sayısı | Etiket (`.txt`) Sayısı | Toplam Kutu (Instances) | Küçük Kutu (`≤ 32×32 px`) | Genel Küçük Oranı (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test** | 2974 | 2974 (97 arka plan) | 9611 | 2295 | %23.88 |

Test kümesinin sınıf bazlı dağılımı aşağıdadır (küçük nesne: COCO tanımıyla alan ≤ 1024 px²):

| Sınıf ID | Sınıf Adı | Örnek Sayısı | Küçük Nesne Sayısı | Sınıf İçi Küçük Oranı (%) |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `person` | 6605 | 1568 | %23.74 |
| 1 | `car` | 1906 | 44 | %2.31 |
| 2 | `bicycle` | 515 | 350 | %67.96 |
| 3 | `other_vehicle` | 19 | 1 | %5.26 |
| 4 | `drone` | 383 | 253 | %66.06 |
| 5 | `mine` | 183 | 79 | %43.17 |
| | **Toplam** | **9611** | **2295** | **%23.88** |


#### Küme Bazında Dağılım (Train / Val / Test)

Gruplar bölünmeden, sınıf dağılımını koruyan katmanlı bölmede hedef oranlar `%80 / %10 / %10` iken, büyük grupların bütünlüğü korunduğu için gerçekleşen oranlar `%77.02 / %11.58 / %11.40` olmuştur. Kaynak karışımı kümeler arasında benzerdir: LLVIP kaynaklı görüntü oranı `train`'de `%57.7`, `val`'de `%65.0`, `test`'te `%65.2`'dir.

| Küme | Görüntü Sayısı | Oran (%) | Toplam Nesne | Küçük Nesne (≤ 32×32 px) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 20.092 | %77.02 | 68.099 | 14.470 (%21.25) |
| **Val** | 3.020 | %11.58 | 9.060 | 2.673 (%29.50) |
| **Test** | 2.974 | %11.40 | 9.611 | 2.295 (%23.88) |
| **Toplam** | 26.086 | %100.0 | 86.770 | 19.438 (%22.40) |

Test kümesindeki 2.974 görüntünün 97'si etiket içermeyen arka plan görüntüsüdür.

> **Val ve test farkı:** `val` kümesinin küçük nesne oranı (`%29.50`) `test` kümesinden (`%23.88`) yüksektir. Bu fark, grupların bütün hâlinde atanmasının kümelerin bileşimini kısmen farklılaştırmasıyla ilişkilidir; farkın tek bir kaynağa veya irtifa grubuna bağlı olduğu gösterilmemiştir (HIT-UAV görüntü sayısı `val` ve `test`'te `283` ve `284`; 130 m dizilerinin payı `%12.4` ve `%9.9`).

---

## 2. Kümeler Arası Veri Sızıntısı (Data Leakage) Kontrolü ve Çözümü

İlk aşamada yapılan MD5 (birebir eşleşme) ve dHash (algısal benzerlik) kontrollerinde tespit edilen sızıntılar giderilmiş, veri seti yeniden yapılandırılmış ve son doğrulama testleri tamamlanmıştır:

- **Önceki Durum ve Çok Aşamalı Kopya Tespiti:** Birebir kopyalar için MD5 özet kontrolü (`2` birebir kopya) ve D4 simetri dönüşümleri (döndürme ve yansıtma) dahil edilerek `dHash (Hamming <= 6)` + `64x64` Normalize Çapraz Korelasyon (`NCC >= 0.95`) ölçütleriyle toplam `764.805` yakın-kopya çifti tespit edilmiş ve Union-Find algoritmasıyla `13.234` bağımsız grupta toplanmıştır.
- **Uygulanan Çözüm:** Adil ve akademik olarak güvenilir bir değerlendirme sağlamak amacıyla bu grupların bütünlüğü bozulmadan katmanlı (`stratified`) bölme uygulanarak tüm veri seti yeniden `train` (`20.092`), `val` (`3.020`) ve `test` (`2.974`) kümelerine ayrılmıştır.
- **Son Doğrulama Sonuçları (MD5 / Hamming / NCC):**
  - **Birden Fazla Kümeye Dağılmış Grup:** `0`
  - **Kümeler Arası Birebir Kopya (MD5):** `0`
  - **Kümeler Arası Yakın Kopya (`Hamming <= 6`, `NCC >= 0.95`):** `0`
  - **Farklı Kümelerde Aynı Dosya Adı:** `0`
  - **Küme Çiftleri Arası Maksimum Benzerlik (NCC, Ham<=10):**
    - `val -> train`: `0.995`
    - `test -> train`: `1.000`
    - `test -> val`: `0.976`
  - **Doğrulama Çıktısı:** `SIZINTI YOK - train/val/test ayrık` (**Bölme Parmak İzi / Split Fingerprint:** `da43dd521edebb0d`)

> **Not:** Daha gevşek bir denetimde (`Hamming <= 10`, `NCC >= 0.90`) `test -> train` NCC değerinin `1.000` çıkması, farklı dizilerde görsel olarak benzer sahnelerin (ör. aynı kamera düzeneği veya sabit arka plan) bulunabildiğini göstermektedir. Bu durum aynı dosya veya aynı uçuş dizisinin kümeler arasında paylaşıldığı anlamına gelmez; belirlenen eşiklerde (`Hamming <= 6`, `NCC >= 0.95`) yukarıdaki MD5, yakın-kopya ve dosya adı kontrolleri `0` sonuç vermiştir.

---

## 3. Doğal Dil Raporlama Modülü

Nesne tespit modellerinden alınan ham çıktıları operatör dostu Türkçe durum raporuna dönüştüren modüldür:

- **Kodun Bulunduğu Dosya:** `report_generator.py` (`generate_report` fonksiyonu)
- **Girdi Formatı:** Güven eşiği (`SAMPLE_CONF = 0.25`) üzerindeki sınıf adları ve tespit adedini içeren tespit listesi (Örn: `['person', 'person', 'car']`)
- **Çıktı Formatı:** Doğal dilde rapor cümlesi (Örn: `"Gözlem sahasında 2 adet person ve 1 adet car tespit edilmiştir."`; tespit yoksa `"Gözlem sahasında herhangi bir nesne tespit edilmemiştir."`)
- **Çalışma Adımları:** (1) tespitler `SAMPLE_CONF = 0.25` güven eşiğiyle süzülür; (2) kalan tespitlerin sınıf adları `Counter` ile sayılır; (3) her sınıf için "N adet sınıf_adı" kalıbı üretilir; (4) birden fazla sınıf varsa kalıplar virgül ve "ve" bağlacıyla birleştirilir; (5) hiç tespit yoksa boş-sahne cümlesi üretilir. Sınıf adları modelin etiketleriyle aynı biçimde, İngilizce (`person`, `car`) bırakılmıştır.
- **Örnek Seçimi:** 20 örnek görüntü test kümesinden `SEED = 42` ile deterministik olarak seçilmiştir (10 görüntü küçük nesne içeren havuzdan, 10 görüntü genel havuzdan).
- **Kısıt:** Modül deterministik ve halüsinasyonsuz özetler üretse de yalnızca güven eşiği üzerindeki sınıf sayılarını aktarmakta, nesneler arasındaki konumsal veya mekânsal ilişkileri (ör. "sol
